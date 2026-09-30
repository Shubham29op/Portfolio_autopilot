"""Broker interface shared by PaperBroker (now) and KiteBroker (later)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Order:
    id: str
    symbol: str
    side: str                 # buy | sell
    qty: int
    reason: str               # entry | model_exit | time_stop | max_hold | stop_unfilled | circuit_breaker | exit_all
    created: str              # ISO date the decision was made
    limit_price: Optional[float] = None   # buy: max price we accept
    stop_atr: Optional[float] = None      # for entries: stop distance in ATR
    atr: Optional[float] = None
    target_atr: Optional[float] = None
    prob: Optional[float] = None
    protective: bool = False  # never blocked by caps or pause
    status: str = "queued"    # queued | filled | cancelled | rejected
    note: str = ""


@dataclass
class GTT:
    """Mirrors a Kite GTT OCO: one stop leg and one target leg, one-shot."""
    id: str
    symbol: str
    qty: int
    stop_trigger: float
    stop_limit: float
    target_trigger: float
    target_limit: float
    status: str = "active"    # active | triggered | cancelled
    pending_leg: Optional[str] = None   # stop leg triggered but limit not filled yet


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    entry_date: str
    atr_at_entry: float
    stop_atr: float
    highest_close: float
    prob_at_entry: Optional[float] = None
    gtt_id: Optional[str] = None
    unprotected: bool = False
    last_price: Optional[float] = None
    entry_charges: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: int
    price: float
    date: str
    reason: str
    charges: dict = field(default_factory=dict)
    pnl: Optional[float] = None       # realised P&L net of both legs' charges (sells only)
