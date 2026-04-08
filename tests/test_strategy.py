from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strategy import TrebBaseHighriskLpStrategy


class MarketStub:
    def __init__(self):
        self._prices = {"TREB": Decimal("1"), "USDC": Decimal("1")}
        self._balances = {
            "TREB": SimpleNamespace(balance=Decimal("1000"), balance_usd=Decimal("1000")),
            "USDC": SimpleNamespace(balance=Decimal("1000"), balance_usd=Decimal("1000")),
        }
        self._ohlcv = [{"high": Decimal("1.015")}, {"high": Decimal("1.01")}]

    def price(self, token: str):
        return self._prices[token]

    def balance(self, token: str):
        return self._balances[token]

    def ohlcv(self, token: str, timeframe: str, limit: int):
        return self._ohlcv


@pytest.fixture
def config() -> dict:
    return {
        "protocol": "uniswap_v3",
        "pool": "TREB/USDC/3000",
        "base_token": "TREB",
        "quote_token": "USDC",
        "entry_dip_min_pct": 0.01,
        "entry_dip_max_pct": 0.02,
        "target_allocation_pct": 0.015,
        "max_allocation_pct": 0.02,
        "stop_loss_pct": 0.30,
        "tp_levels_pct": [0.25, 0.50, 1.00],
        "tp_reopen_allocations_pct": [0.01, 0.005, 0.0],
        "range_width_pct": 0.10,
        "rebalance_threshold_pct": 0.85,
        "max_slippage": 0.01,
        "ohlcv_timeframe": "15m",
        "ohlcv_lookback": 96,
        "min_quote_reserve_usd": 50,
    }


@pytest.fixture
def strategy(config: dict) -> TrebBaseHighriskLpStrategy:
    return TrebBaseHighriskLpStrategy(
        config=config,
        chain="base",
        wallet_address="0x" + "1" * 40,
    )


@pytest.fixture
def market() -> MarketStub:
    return MarketStub()


def test_holds_when_price_unavailable(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    market.price = MagicMock(side_effect=ValueError("price unavailable"))
    result = strategy.decide(market)
    assert result.intent_type.value == "HOLD"
    assert result.reason_code == "DATA_UNAVAILABLE"


def test_opens_lp_when_dip_is_in_band(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    result = strategy.decide(market)
    assert result.intent_type.value == "LP_OPEN"
    assert result.pool == "TREB/USDC/3000"
    assert result.protocol == "uniswap_v3"
    assert result.amount0 > 0
    assert result.amount1 > 0


def test_holds_when_dip_condition_not_met(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    market._ohlcv = [{"high": Decimal("1.004")}, {"high": Decimal("1.003")}]
    result = strategy.decide(market)
    assert result.intent_type.value == "HOLD"
    assert result.reason_code == "ENTRY_CONDITION_NOT_MET"


def test_holds_on_insufficient_quote_balance(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    market._balances["USDC"] = SimpleNamespace(balance=Decimal("10"), balance_usd=Decimal("10"))
    result = strategy.decide(market)
    assert result.intent_type.value == "HOLD"
    assert result.reason_code == "INSUFFICIENT_BALANCE"


def test_stop_loss_closes_position(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    strategy._position_id = "123"
    strategy._entry_price = Decimal("1")
    market._prices["TREB"] = Decimal("0.69")
    result = strategy.decide(market)
    assert result.intent_type.value == "LP_CLOSE"
    assert strategy._pending_close_reason == "STOP_LOSS"


def test_take_profit_25_closes_position(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    strategy._position_id = "123"
    strategy._entry_price = Decimal("1")
    market._prices["TREB"] = Decimal("1.3")
    result = strategy.decide(market)
    assert result.intent_type.value == "LP_CLOSE"
    assert strategy._pending_reopen_allocation_pct == Decimal("0.01")


def test_take_profit_100_closes_and_no_reopen(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    strategy._position_id = "123"
    strategy._entry_price = Decimal("1")
    strategy._tp_hits[str(Decimal("0.25"))] = True
    strategy._tp_hits[str(Decimal("0.5"))] = True
    market._prices["TREB"] = Decimal("2.2")
    result = strategy.decide(market)
    assert result.intent_type.value == "LP_CLOSE"
    assert strategy._pending_reopen_allocation_pct == Decimal("0")


def test_rebalance_close_when_price_out_of_range(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    strategy._position_id = "123"
    strategy._entry_price = Decimal("1")
    strategy._current_allocation_pct = Decimal("0.015")
    strategy._range_lower = Decimal("0.95")
    strategy._range_upper = Decimal("1.05")
    market._prices["TREB"] = Decimal("1.20")
    result = strategy.decide(market)
    assert result.intent_type.value == "LP_CLOSE"
    assert strategy._pending_close_reason == "RANGE_REBALANCE"


def test_pending_reopen_opens_without_dip_check(strategy: TrebBaseHighriskLpStrategy, market: MarketStub):
    strategy._pending_reopen_allocation_pct = Decimal("0.005")
    market._ohlcv = []
    result = strategy.decide(market)
    assert result.intent_type.value == "LP_OPEN"


def test_on_intent_executed_updates_open_state(strategy: TrebBaseHighriskLpStrategy):
    strategy._pending_open_alloc_pct = Decimal("0.01")
    open_intent = SimpleNamespace(
        intent_type=SimpleNamespace(value="LP_OPEN"),
        range_lower=Decimal("0.95"),
        range_upper=Decimal("1.05"),
    )
    result = SimpleNamespace(position_id="999")
    strategy.on_intent_executed(open_intent, True, result)
    assert strategy._position_id == "999"
    assert strategy._entry_price == Decimal("1.00")
    assert strategy._current_allocation_pct == Decimal("0.01")


def test_teardown_intents_order(strategy: TrebBaseHighriskLpStrategy):
    from almanak.framework.teardown import TeardownMode

    strategy._position_id = "123"
    intents = strategy.generate_teardown_intents(TeardownMode.SOFT)
    assert len(intents) == 2
    assert intents[0].intent_type.value == "LP_CLOSE"
    assert intents[1].intent_type.value == "SWAP"


def test_persistent_state_roundtrip(strategy: TrebBaseHighriskLpStrategy, config: dict):
    strategy._position_id = "123"
    strategy._entry_price = Decimal("1.2")
    strategy._range_lower = Decimal("1.1")
    strategy._range_upper = Decimal("1.3")
    strategy._current_allocation_pct = Decimal("0.01")
    strategy._pending_reopen_allocation_pct = Decimal("0.005")
    strategy._tp_hits[str(Decimal("0.25"))] = True

    state = strategy.get_persistent_state()

    restored = TrebBaseHighriskLpStrategy(
        config=config,
        chain="base",
        wallet_address="0x" + "2" * 40,
    )
    restored.load_persistent_state(state)

    assert restored._position_id == "123"
    assert restored._entry_price == Decimal("1.2")
    assert restored._range_lower == Decimal("1.1")
    assert restored._range_upper == Decimal("1.3")
    assert restored._current_allocation_pct == Decimal("0.01")
    assert restored._pending_reopen_allocation_pct == Decimal("0.005")
    assert restored._tp_hits[str(Decimal("0.25"))] is True
