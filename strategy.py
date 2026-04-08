"""TREB high-risk LP strategy mapped to Uniswap V3 on Base."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

from almanak.framework.intents import Intent
from almanak.framework.strategies import IntentStrategy, MarketSnapshot, almanak_strategy

logger = logging.getLogger(__name__)


@almanak_strategy(
    name="treb_base_highrisk_lp",
    description="Risk-managed TREB LP strategy on Uniswap V3 Base",
    version="1.0.0",
    author="Almanak",
    tags=["lp", "uniswap_v3", "base", "treb", "risk-managed"],
    supported_chains=["base"],
    supported_protocols=["uniswap_v3"],
    intent_types=["LP_OPEN", "LP_CLOSE", "SWAP", "HOLD"],
    default_chain="base",
)
class TrebBaseHighriskLpStrategy(IntentStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def get_cfg(key: str, default: Any) -> Any:
            if isinstance(self.config, dict):
                return self.config.get(key, default)
            return getattr(self.config, key, default)

        self.protocol = get_cfg("protocol", "uniswap_v3")
        self.pool = get_cfg("pool", "TREB/USDC/3000")
        self.base_token = get_cfg("base_token", "TREB")
        self.quote_token = get_cfg("quote_token", "USDC")

        self.entry_dip_min_pct = self._to_pct(get_cfg("entry_dip_min_pct", "0.01"))
        self.entry_dip_max_pct = self._to_pct(get_cfg("entry_dip_max_pct", "0.02"))
        self.target_allocation_pct = self._to_pct(get_cfg("target_allocation_pct", "0.015"))
        self.max_allocation_pct = self._to_pct(get_cfg("max_allocation_pct", "0.02"))
        self.stop_loss_pct = self._to_pct(get_cfg("stop_loss_pct", "0.30"))
        self.range_width_pct = self._to_pct(get_cfg("range_width_pct", "0.10"))
        self.rebalance_threshold_pct = self._to_pct(get_cfg("rebalance_threshold_pct", "0.85"))
        self.max_slippage = self._to_pct(get_cfg("max_slippage", "0.01"))

        self.ohlcv_timeframe = get_cfg("ohlcv_timeframe", "15m")
        self.ohlcv_lookback = int(get_cfg("ohlcv_lookback", 96))
        self.min_quote_reserve_usd = Decimal(str(get_cfg("min_quote_reserve_usd", "50")))

        raw_tp_levels = get_cfg("tp_levels_pct", [0.25, 0.50, 1.00])
        raw_tp_reopen = get_cfg("tp_reopen_allocations_pct", [0.01, 0.005, 0.0])
        self.tp_levels_pct = [self._to_pct(v) for v in raw_tp_levels]
        self.tp_reopen_allocations_pct = [self._to_pct(v) for v in raw_tp_reopen]
        self._tp_to_reopen = {
            str(level): self.tp_reopen_allocations_pct[idx]
            for idx, level in enumerate(self.tp_levels_pct)
            if idx < len(self.tp_reopen_allocations_pct)
        }

        self._position_id: str | None = None
        self._entry_price: Decimal | None = None
        self._range_lower: Decimal | None = None
        self._range_upper: Decimal | None = None
        self._current_allocation_pct: Decimal = Decimal("0")
        self._pending_reopen_allocation_pct: Decimal = Decimal("0")
        self._tp_hits: dict[str, bool] = {str(level): False for level in self.tp_levels_pct}
        self._last_action: str = "init"
        self._last_reason_code: str | None = None
        self._pending_close_reason: str | None = None
        self._pending_open_alloc_pct: Decimal = Decimal("0")

    @staticmethod
    def _to_pct(value: Any) -> Decimal:
        pct = Decimal(str(value))
        if pct > Decimal("1"):
            return pct / Decimal("100")
        return pct

    @staticmethod
    def _extract_balance_and_usd(balance_obj: Any, token_price: Decimal) -> tuple[Decimal, Decimal]:
        if isinstance(balance_obj, Decimal):
            balance = balance_obj
            return balance, balance * token_price
        if hasattr(balance_obj, "balance"):
            balance = Decimal(str(balance_obj.balance))
            balance_usd_attr = getattr(balance_obj, "balance_usd", None)
            if balance_usd_attr is not None:
                return balance, Decimal(str(balance_usd_attr))
            return balance, balance * token_price
        balance = Decimal(str(balance_obj))
        return balance, balance * token_price

    def _safe_hold(self, reason: str, reason_code: str, details: dict[str, Any] | None = None) -> Intent:
        self._last_action = "hold"
        self._last_reason_code = reason_code
        return Intent.hold(reason=reason, reason_code=reason_code, reason_details=details or {}, chain=self.chain)

    def _extract_recent_high(self, ohlcv_data: Any) -> Decimal:
        if ohlcv_data is None:
            raise ValueError("OHLCV data unavailable")
        if hasattr(ohlcv_data, "empty") and ohlcv_data.empty:
            raise ValueError("OHLCV data empty")
        if isinstance(ohlcv_data, list) and ohlcv_data:
            highs = [Decimal(str(row["high"])) for row in ohlcv_data if isinstance(row, dict) and "high" in row]
            if highs:
                return max(highs)
        if hasattr(ohlcv_data, "__getitem__"):
            highs = ohlcv_data["high"]
            if hasattr(highs, "max"):
                return Decimal(str(highs.max()))
            if isinstance(highs, list) and highs:
                return max(Decimal(str(v)) for v in highs)
        raise ValueError("Unable to compute recent high")

    def _build_lp_open(self, current_price: Decimal, allocation_pct: Decimal, base_balance: Decimal, quote_balance: Decimal) -> Intent:
        portfolio_usd = quote_balance + (base_balance * current_price)
        allocation_pct = min(allocation_pct, self.max_allocation_pct)
        allocation_usd = portfolio_usd * allocation_pct
        each_side_usd = allocation_usd / Decimal("2")

        base_amount = each_side_usd / current_price if current_price > 0 else Decimal("0")
        quote_amount = each_side_usd

        if allocation_usd <= 0 or base_amount <= 0 or quote_amount <= 0:
            return self._safe_hold("Computed LP size is zero", "SIZE_ZERO")

        if base_balance < base_amount:
            return self._safe_hold(
                f"Insufficient {self.base_token} balance",
                "INSUFFICIENT_BALANCE",
                {"required": str(base_amount), "available": str(base_balance)},
            )

        if quote_balance < quote_amount + self.min_quote_reserve_usd:
            return self._safe_hold(
                f"Insufficient {self.quote_token} reserve",
                "INSUFFICIENT_BALANCE",
                {
                    "required_quote": str(quote_amount + self.min_quote_reserve_usd),
                    "available_quote": str(quote_balance),
                },
            )

        half_range = self.range_width_pct / Decimal("2")
        range_lower = current_price * (Decimal("1") - half_range)
        range_upper = current_price * (Decimal("1") + half_range)

        self._pending_open_alloc_pct = allocation_pct
        self._last_action = "lp_open"
        self._last_reason_code = None
        return Intent.lp_open(
            pool=self.pool,
            amount0=base_amount,
            amount1=quote_amount,
            range_lower=range_lower,
            range_upper=range_upper,
            protocol=self.protocol,
            chain=self.chain,
        )

    def _build_lp_close(self, reason_code: str, reason: str, pending_reopen: Decimal = Decimal("0")) -> Intent:
        self._pending_reopen_allocation_pct = pending_reopen
        self._pending_close_reason = reason_code
        self._last_action = "lp_close"
        self._last_reason_code = reason_code
        return Intent.lp_close(
            position_id=str(self._position_id),
            pool=self.pool,
            collect_fees=True,
            protocol=self.protocol,
            chain=self.chain,
        )

    def decide(self, market: MarketSnapshot) -> Optional[Intent]:
        try:
            base_price = Decimal(str(market.price(self.base_token)))
            quote_price = Decimal(str(market.price(self.quote_token)))
            base_balance_obj = market.balance(self.base_token)
            quote_balance_obj = market.balance(self.quote_token)
            base_balance, base_balance_usd = self._extract_balance_and_usd(base_balance_obj, base_price)
            quote_balance, quote_balance_usd = self._extract_balance_and_usd(quote_balance_obj, quote_price)
        except Exception as exc:
            logger.exception("Market data fetch failed")
            return self._safe_hold(
                "Market data unavailable",
                "DATA_UNAVAILABLE",
                {"error": str(exc)},
            )

        current_price = base_price / quote_price if quote_price > 0 else base_price

        if self._position_id:
            if not self._entry_price or self._entry_price <= 0:
                return self._safe_hold("Missing entry price for open position", "STATE_INVALID")

            pnl_pct = (current_price - self._entry_price) / self._entry_price
            if pnl_pct <= -self.stop_loss_pct:
                return self._build_lp_close("STOP_LOSS", "Stop-loss triggered", Decimal("0"))

            for level in sorted(self.tp_levels_pct, reverse=True):
                level_key = str(level)
                if pnl_pct >= level and not self._tp_hits.get(level_key, False):
                    reopen_alloc = self._tp_to_reopen.get(level_key, Decimal("0"))
                    return self._build_lp_close("TAKE_PROFIT", f"Take-profit {level_key} reached", reopen_alloc)

            if self._range_lower is not None and self._range_upper is not None and self._range_upper > self._range_lower:
                range_size = self._range_upper - self._range_lower
                position_in_range = (current_price - self._range_lower) / range_size
                lower_bound = (Decimal("1") - self.rebalance_threshold_pct) / Decimal("2")
                upper_bound = (Decimal("1") + self.rebalance_threshold_pct) / Decimal("2")
                if position_in_range < lower_bound or position_in_range > upper_bound:
                    return self._build_lp_close(
                        "RANGE_REBALANCE",
                        "Price moved outside rebalance bounds",
                        self._current_allocation_pct,
                    )

            return self._safe_hold("Position active; no exit signal", "POSITION_ACTIVE", {"pnl_pct": str(pnl_pct)})

        reopen_pct = self._pending_reopen_allocation_pct
        if reopen_pct > 0:
            return self._build_lp_open(current_price, reopen_pct, base_balance, quote_balance)

        try:
            ohlcv = market.ohlcv(self.base_token, timeframe=self.ohlcv_timeframe, limit=self.ohlcv_lookback)
            recent_high = self._extract_recent_high(ohlcv)
        except Exception as exc:
            return self._safe_hold("Dip signal unavailable", "DATA_UNAVAILABLE", {"error": str(exc)})

        if recent_high <= 0:
            return self._safe_hold("Invalid recent high", "DATA_UNAVAILABLE")

        dip_pct = (recent_high - current_price) / recent_high
        if dip_pct < self.entry_dip_min_pct or dip_pct > self.entry_dip_max_pct:
            return self._safe_hold(
                "Waiting for entry dip",
                "ENTRY_CONDITION_NOT_MET",
                {"dip_pct": str(dip_pct)},
            )

        portfolio_usd = base_balance_usd + quote_balance_usd
        if portfolio_usd <= 0:
            return self._safe_hold("Portfolio value unavailable", "DATA_UNAVAILABLE")

        alloc_pct = min(self.target_allocation_pct, self.max_allocation_pct)
        return self._build_lp_open(current_price, alloc_pct, base_balance, quote_balance)

    def on_intent_executed(self, intent, success: bool, result):
        if not success:
            return

        intent_type = getattr(getattr(intent, "intent_type", None), "value", "")
        if intent_type == "LP_OPEN":
            result_position_id = getattr(result, "position_id", None) if result else None
            if result_position_id is not None:
                self._position_id = str(result_position_id)
            self._entry_price = getattr(intent, "range_lower", None)
            if self._entry_price and self._range_upper:
                pass
            open_lower = getattr(intent, "range_lower", None)
            open_upper = getattr(intent, "range_upper", None)
            if open_lower is not None:
                self._range_lower = Decimal(str(open_lower))
            if open_upper is not None:
                self._range_upper = Decimal(str(open_upper))
            if self._range_lower is not None and self._range_upper is not None:
                self._entry_price = (self._range_lower + self._range_upper) / Decimal("2")
            self._current_allocation_pct = self._pending_open_alloc_pct
            self._pending_open_alloc_pct = Decimal("0")
            self._pending_reopen_allocation_pct = Decimal("0")

        elif intent_type == "LP_CLOSE":
            if self._pending_close_reason == "TAKE_PROFIT" and self._entry_price is not None:
                close_reopen = self._pending_reopen_allocation_pct
                for level in sorted(self.tp_levels_pct):
                    key = str(level)
                    if close_reopen == self._tp_to_reopen.get(key, Decimal("0")):
                        self._tp_hits[key] = True
            self._position_id = None
            self._entry_price = None
            self._range_lower = None
            self._range_upper = None
            self._current_allocation_pct = Decimal("0")
            self._pending_close_reason = None

    def supports_teardown(self) -> bool:
        return True

    def get_open_positions(self):
        from almanak.framework.teardown import PositionInfo, PositionType, TeardownPositionSummary

        positions: list[PositionInfo] = []
        if self._position_id:
            estimated_value = Decimal("0")
            if self._entry_price and self._current_allocation_pct > 0:
                estimated_value = self._entry_price * self._current_allocation_pct
            positions.append(
                PositionInfo(
                    position_type=PositionType.LP,
                    position_id=str(self._position_id),
                    chain=self.chain,
                    protocol=self.protocol,
                    value_usd=estimated_value,
                    details={
                        "pool": self.pool,
                        "range_lower": str(self._range_lower) if self._range_lower is not None else None,
                        "range_upper": str(self._range_upper) if self._range_upper is not None else None,
                        "allocation_pct": str(self._current_allocation_pct),
                    },
                )
            )

        return TeardownPositionSummary(
            strategy_id=getattr(self, "strategy_id", "treb_base_highrisk_lp"),
            timestamp=datetime.now(UTC),
            positions=positions,
        )

    def generate_teardown_intents(self, mode, market=None) -> list[Intent]:
        from almanak.framework.teardown import TeardownMode

        if not self._position_id:
            return []

        max_slippage = Decimal("0.03") if mode == TeardownMode.HARD else Decimal("0.005")
        return [
            Intent.lp_close(
                position_id=str(self._position_id),
                pool=self.pool,
                collect_fees=True,
                protocol=self.protocol,
                chain=self.chain,
            ),
            Intent.swap(
                from_token=self.base_token,
                to_token=self.quote_token,
                amount="all",
                max_slippage=max_slippage,
                protocol=self.protocol,
                chain=self.chain,
            ),
        ]

    def get_persistent_state(self) -> dict[str, Any]:
        return {
            "position_id": self._position_id,
            "entry_price": str(self._entry_price) if self._entry_price is not None else None,
            "range_lower": str(self._range_lower) if self._range_lower is not None else None,
            "range_upper": str(self._range_upper) if self._range_upper is not None else None,
            "current_allocation_pct": str(self._current_allocation_pct),
            "pending_reopen_allocation_pct": str(self._pending_reopen_allocation_pct),
            "tp_hits": self._tp_hits,
            "last_action": self._last_action,
            "last_reason_code": self._last_reason_code,
        }

    def load_persistent_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self._position_id = state.get("position_id")
        entry_price = state.get("entry_price")
        self._entry_price = Decimal(entry_price) if entry_price else None
        range_lower = state.get("range_lower")
        self._range_lower = Decimal(range_lower) if range_lower else None
        range_upper = state.get("range_upper")
        self._range_upper = Decimal(range_upper) if range_upper else None
        self._current_allocation_pct = Decimal(str(state.get("current_allocation_pct", "0")))
        self._pending_reopen_allocation_pct = Decimal(str(state.get("pending_reopen_allocation_pct", "0")))

        state_tp_hits = state.get("tp_hits", {})
        if isinstance(state_tp_hits, dict):
            for key in self._tp_hits:
                self._tp_hits[key] = bool(state_tp_hits.get(key, False))

        self._last_action = str(state.get("last_action", self._last_action))
        self._last_reason_code = state.get("last_reason_code")

    def get_status(self) -> dict[str, Any]:
        return {
            "strategy": "treb_base_highrisk_lp",
            "chain": self.chain,
            "pool": self.pool,
            "position_id": self._position_id,
            "entry_price": str(self._entry_price) if self._entry_price is not None else None,
            "allocation_pct": str(self._current_allocation_pct),
            "pending_reopen_allocation_pct": str(self._pending_reopen_allocation_pct),
            "tp_hits": self._tp_hits,
            "last_action": self._last_action,
            "last_reason_code": self._last_reason_code,
        }
