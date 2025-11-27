# -*- coding: utf-8 -*-
"""
adapters/alpaca/order_execution.py
Alpaca order execution adapter for US equities.

Handles order submission, cancellation, and position tracking.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core_models import Order, ExecReport, Position, Side, OrderType, ExecStatus, Liquidity
from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import AccountInfo, ExchangeVendor

logger = logging.getLogger(__name__)


class AlpacaOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Alpaca order execution adapter for US equities.

    Handles order placement, cancellation, and position management.

    Configuration:
        api_key: Alpaca API key (required)
        api_secret: Alpaca API secret (required)
        paper: Use paper trading endpoint (default: True)
        extended_hours: Allow extended hours trading (default: False)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._client = None

    def _get_client(self):
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError:
                raise ImportError(
                    "Alpaca SDK not installed. Install with: pip install alpaca-py"
                )

            api_key = self._config.get("api_key")
            api_secret = self._config.get("api_secret")
            paper = self._config.get("paper", True)

            if not api_key or not api_secret:
                raise ValueError("Alpaca API key and secret are required")

            self._client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=paper,
            )

        return self._client

    def _do_connect(self) -> None:
        """Initialize client connection."""
        self._get_client()

    def _do_disconnect(self) -> None:
        """Close client connection."""
        self._client = None

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order to Alpaca.

        Args:
            order: Order to submit

        Returns:
            OrderResult with status
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce

            # Convert side
            side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL

            # Convert time in force
            tif_map = {
                "GTC": TimeInForce.GTC,
                "IOC": TimeInForce.IOC,
                "FOK": TimeInForce.FOK,
                "DAY": TimeInForce.DAY,
            }
            tif = tif_map.get(order.time_in_force.value, TimeInForce.DAY)

            # Extended hours
            extended = self._config.get("extended_hours", False)

            # Create order request
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=side,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                    extended_hours=extended,
                )
            else:  # LIMIT
                if order.price is None:
                    return OrderResult(
                        success=False,
                        error_code="MISSING_PRICE",
                        error_message="Limit orders require a price",
                    )

                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=float(order.quantity),
                    side=side,
                    time_in_force=tif,
                    limit_price=float(order.price),
                    client_order_id=order.client_order_id,
                    extended_hours=extended,
                )

            # Submit order
            alpaca_order = client.submit_order(request)

            # Parse response
            return OrderResult(
                success=True,
                order_id=str(alpaca_order.id),
                client_order_id=str(alpaca_order.client_order_id),
                status=str(alpaca_order.status.value),
                filled_qty=Decimal(str(alpaca_order.filled_qty or 0)),
                filled_price=Decimal(str(alpaca_order.filled_avg_price))
                if alpaca_order.filled_avg_price
                else None,
                raw_response={"id": str(alpaca_order.id)},
            )

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return OrderResult(
                success=False,
                error_code="SUBMISSION_FAILED",
                error_message=str(e),
            )

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Alpaca order ID
            client_order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        try:
            client = self._get_client()

            if order_id:
                client.cancel_order_by_id(order_id)
            elif client_order_id:
                # Need to find order by client_order_id
                orders = client.get_orders()
                for o in orders:
                    if str(o.client_order_id) == client_order_id:
                        client.cancel_order_by_id(str(o.id))
                        return True
                logger.warning(f"Order with client_order_id {client_order_id} not found")
                return False
            else:
                raise ValueError("Either order_id or client_order_id required")

            return True

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: Alpaca order ID
            client_order_id: Client order ID

        Returns:
            ExecReport or None
        """
        try:
            client = self._get_client()

            if order_id:
                alpaca_order = client.get_order_by_id(order_id)
            elif client_order_id:
                alpaca_order = client.get_order_by_client_id(client_order_id)
            else:
                return None

            return self._convert_order_to_exec_report(alpaca_order)

        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
            return None

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol

        Returns:
            List of open orders
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            if symbol:
                request.symbols = [symbol]

            alpaca_orders = client.get_orders(request)

            orders = []
            for o in alpaca_orders:
                order = self._convert_alpaca_order(o)
                if order:
                    orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Position]:
        """
        Get current positions.

        Args:
            symbols: Filter by symbols

        Returns:
            Dict mapping symbol to Position
        """
        try:
            client = self._get_client()
            alpaca_positions = client.get_all_positions()

            positions = {}
            for p in alpaca_positions:
                if symbols and p.symbol not in symbols:
                    continue

                import time
                position = Position(
                    symbol=p.symbol,
                    qty=Decimal(str(p.qty)),
                    avg_entry_price=Decimal(str(p.avg_entry_price)),
                    realized_pnl=Decimal("0"),  # Alpaca doesn't provide this directly
                    ts=int(time.time() * 1000),
                    meta={
                        "market_value": str(p.market_value),
                        "cost_basis": str(p.cost_basis),
                        "unrealized_pl": str(p.unrealized_pl),
                        "unrealized_plpc": str(p.unrealized_plpc),
                        "current_price": str(p.current_price),
                        "side": str(p.side),
                    },
                )
                positions[p.symbol] = position

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status
        """
        try:
            client = self._get_client()
            account = client.get_account()

            return AccountInfo(
                vendor=self._vendor,
                account_id=str(account.id),
                account_type="margin" if account.multiplier > 1 else "cash",
                vip_tier=0,  # Alpaca doesn't have VIP tiers
                maker_fee_rate=0.0,
                taker_fee_rate=0.0,
                buying_power=Decimal(str(account.buying_power)),
                cash_balance=Decimal(str(account.cash)),
                margin_enabled=account.multiplier > 1,
                pattern_day_trader=account.pattern_day_trader,
                raw_data={
                    "equity": str(account.equity),
                    "portfolio_value": str(account.portfolio_value),
                    "multiplier": str(account.multiplier),
                    "status": str(account.status),
                    "daytrade_count": account.daytrade_count,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return AccountInfo(
                vendor=self._vendor,
                raw_data={"error": str(e)},
            )

    def _convert_alpaca_order(self, alpaca_order: Any) -> Optional[Order]:
        """Convert Alpaca order to our Order model."""
        try:
            from core_models import TimeInForce

            side = Side.BUY if str(alpaca_order.side) == "buy" else Side.SELL
            order_type = (
                OrderType.MARKET
                if str(alpaca_order.type) == "market"
                else OrderType.LIMIT
            )

            tif_map = {
                "day": TimeInForce.GTC,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }

            return Order(
                ts=int(alpaca_order.created_at.timestamp() * 1000),
                symbol=alpaca_order.symbol,
                side=side,
                order_type=order_type,
                quantity=Decimal(str(alpaca_order.qty)),
                price=Decimal(str(alpaca_order.limit_price))
                if alpaca_order.limit_price
                else None,
                time_in_force=tif_map.get(str(alpaca_order.time_in_force), TimeInForce.GTC),
                client_order_id=str(alpaca_order.client_order_id),
                meta={"alpaca_id": str(alpaca_order.id)},
            )
        except Exception as e:
            logger.debug(f"Order conversion error: {e}")
            return None

    def _convert_order_to_exec_report(self, alpaca_order: Any) -> ExecReport:
        """Convert Alpaca order to ExecReport."""
        import time

        side = Side.BUY if str(alpaca_order.side) == "buy" else Side.SELL
        order_type = (
            OrderType.MARKET
            if str(alpaca_order.type) == "market"
            else OrderType.LIMIT
        )

        # Map status
        status_map = {
            "new": ExecStatus.NEW,
            "partially_filled": ExecStatus.PARTIALLY_FILLED,
            "filled": ExecStatus.FILLED,
            "canceled": ExecStatus.CANCELED,
            "expired": ExecStatus.CANCELED,
            "rejected": ExecStatus.REJECTED,
            "pending_new": ExecStatus.NEW,
            "accepted": ExecStatus.NEW,
        }
        status = status_map.get(str(alpaca_order.status), ExecStatus.NEW)

        return ExecReport(
            ts=int(time.time() * 1000),
            run_id="",
            symbol=alpaca_order.symbol,
            side=side,
            order_type=order_type,
            price=Decimal(str(alpaca_order.filled_avg_price or alpaca_order.limit_price or 0)),
            quantity=Decimal(str(alpaca_order.filled_qty or 0)),
            fee=Decimal("0"),  # Alpaca is commission-free
            fee_asset="USD",
            exec_status=status,
            liquidity=Liquidity.UNKNOWN,
            client_order_id=str(alpaca_order.client_order_id),
            order_id=str(alpaca_order.id),
            meta={
                "original_qty": str(alpaca_order.qty),
                "status": str(alpaca_order.status),
            },
        )

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Filter by symbol

        Returns:
            Number of orders cancelled
        """
        try:
            client = self._get_client()

            if symbol:
                # Cancel orders for specific symbol
                orders = self.get_open_orders(symbol)
                cancelled = 0
                for order in orders:
                    if self.cancel_order(client_order_id=order.client_order_id):
                        cancelled += 1
                return cancelled
            else:
                # Cancel all orders
                client.cancel_orders()
                return -1  # Unknown count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0
