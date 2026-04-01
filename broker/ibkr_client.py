"""
IBKR Gateway client for the bond-trading strategy.

Connects to a running IBKR Gateway (or TWS), fetches account state,
computes rebalance orders from effective_weights(), and submits them.

Configuration via environment variables:
  IBKR_HOST          Gateway hostname (default: 127.0.0.1)
  IBKR_PORT          Gateway port — 4002 = paper, 4001 = live (default: 4002)
  IBKR_CLIENT_ID     IB API client ID, must be unique per connection (default: 1)
  IBKR_MIN_ORDER_USD Skip orders smaller than this value in USD (default: 50)
"""
import os
import logging
from collections import namedtuple

from ib_insync import IB, Stock, MarketOrder, util

logger = logging.getLogger(__name__)

# One order line returned by build_rebalance_orders()
OrderLine = namedtuple(
    "OrderLine",
    ["ticker", "action", "shares", "est_usd", "current_pct", "target_pct"],
)


class IBKRClient:
    def __init__(self):
        self.host = os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(os.environ.get("IBKR_PORT", "4002"))
        self.client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self.min_order_usd = float(os.environ.get("IBKR_MIN_ORDER_USD", "50"))
        self.ib = IB()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        env = "paper" if self.port == 4002 else "LIVE"
        logger.info(
            "Connecting to IBKR Gateway (%s) at %s:%d  client_id=%d",
            env, self.host, self.port, self.client_id,
        )
        util.patchAsyncio()
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        accounts = self.ib.managedAccounts()
        logger.info("Connected. Managed accounts: %s", accounts)

    def disconnect(self) -> None:
        self.ib.disconnect()
        logger.info("Disconnected from IBKR Gateway.")

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    def get_net_liq(self) -> float:
        """Return account net liquidation value in USD."""
        for v in self.ib.accountValues():
            if v.tag == "NetLiquidation" and v.currency == "USD":
                return float(v.value)
        raise RuntimeError(
            "Could not find NetLiquidation in account values. "
            "Make sure the Gateway is connected and the account is funded."
        )

    def get_positions(self) -> dict:
        """Return {ticker: shares_held} for all current positions."""
        positions = {}
        for pos in self.ib.positions():
            ticker = pos.contract.symbol
            positions[ticker] = int(pos.position)
        return positions

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def get_prices(self, tickers: list) -> dict:
        """
        Fetch market prices for a list of ETF tickers in one snapshot request.
        Returns {ticker: price}. Tickers with no price are omitted with a warning.
        Falls back to close price when live feed is unavailable (market closed).
        """
        contracts = [Stock(t, "SMART", "USD") for t in tickers]
        self.ib.qualifyContracts(*contracts)

        # Use delayed data (type 3) — free for paper accounts; live (type 1) requires subscription.
        # Type 4 = delayed-frozen: also works when market is closed.
        self.ib.reqMarketDataType(4)
        ticker_objs = self.ib.reqTickers(*contracts)

        prices = {}
        for td in ticker_objs:
            symbol = td.contract.symbol
            price = td.marketPrice()
            if price and price == price and price > 0:  # not NaN, not zero
                prices[symbol] = price
            else:
                # marketPrice() returns NaN when market is closed; fall back to close
                close = td.close
                if close and close == close and close > 0:
                    prices[symbol] = close
                    logger.debug("Using close price for %s: %.4f", symbol, close)
                else:
                    logger.warning("No price available for %s — will skip its orders", symbol)

        return prices

    # ------------------------------------------------------------------
    # Order building
    # ------------------------------------------------------------------

    def build_rebalance_orders(
        self,
        target_weights: "pd.Series",
        net_liq: float,
        current_shares: dict,
        prices: dict,
    ) -> list:
        """
        Compute the delta between current positions and target weights.
        Returns a list of OrderLine namedtuples, sorted by abs(est_usd) desc.
        Orders smaller than IBKR_MIN_ORDER_USD are dropped.
        """
        all_tickers = set(target_weights.index) | set(current_shares.keys())
        orders = []

        for ticker in sorted(all_tickers):
            price = prices.get(ticker)
            if not price:
                continue

            target_w = float(target_weights.get(ticker, 0.0))
            target_sh = round((target_w * net_liq) / price)
            current_sh = current_shares.get(ticker, 0)
            delta_sh = target_sh - current_sh

            if delta_sh == 0:
                continue

            est_usd_signed = delta_sh * price
            if abs(est_usd_signed) < self.min_order_usd:
                logger.debug(
                    "Skipping %s: |Δ$| %.0f < min %.0f",
                    ticker, abs(est_usd_signed), self.min_order_usd,
                )
                continue

            action = "BUY" if delta_sh > 0 else "SELL"
            current_pct = (current_sh * price) / net_liq if net_liq else 0.0

            orders.append(OrderLine(
                ticker=ticker,
                action=action,
                shares=abs(delta_sh),
                est_usd=est_usd_signed,
                current_pct=current_pct,
                target_pct=target_w,
            ))

        return sorted(orders, key=lambda o: abs(o.est_usd), reverse=True)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_preview(self, orders: list, net_liq: float) -> None:
        env = "paper" if self.port == 4002 else "LIVE"
        print(f"\n{'━'*64}")
        print(f"ORDER PREVIEW  ({env} account — net liq: ${net_liq:,.0f})")
        print(f"{'━'*64}")
        print(f"  {'Ticker':<8} {'Current%':>9} {'Target%':>8} {'Δ Shares':>9} {'Est. $':>11}  Action")
        print(f"  {'─'*60}")

        total_turnover = 0.0
        for o in orders:
            sign = "+" if o.action == "BUY" else "-"
            delta_str = f"{sign}{o.shares}"
            est_str = f"{sign}${abs(o.est_usd):,.0f}"
            print(
                f"  {o.ticker:<8} {o.current_pct:>8.1%}  {o.target_pct:>7.1%}"
                f"  {delta_str:>9}  {est_str:>11}  {o.action}"
            )
            total_turnover += abs(o.est_usd)

        print(f"  {'─'*60}")
        pct_nav = total_turnover / net_liq if net_liq else 0.0
        print(f"  {len(orders)} order(s)  |  est. turnover: ${total_turnover:,.0f} ({pct_nav:.1%} of NAV)")
        print(f"{'━'*64}")

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_orders(self, orders: list) -> None:
        """Place MarketOrders for each OrderLine. Waits 2s for acknowledgements."""
        if not orders:
            print("No orders to submit.")
            return

        trades = []
        for o in orders:
            contract = Stock(o.ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)
            order = MarketOrder(o.action, o.shares)
            trade = self.ib.placeOrder(contract, order)
            trades.append((o.ticker, trade))
            logger.info("Placed %s %d %s (est. $%.0f)", o.action, o.shares, o.ticker, abs(o.est_usd))

        # Brief pause for order acknowledgements
        self.ib.sleep(2)

        print(f"\n{len(trades)} order(s) submitted:")
        for ticker, trade in trades:
            status = trade.orderStatus.status or "Submitted"
            print(f"  {ticker}: {status}")
