"""Rule guardrails around the model.

Pre-inference rules run before the model (hard business facts, e.g. a blocklisted card).
Post-inference rules can only make a decision MORE cautious, never less, so a model
bug can't approve something the rules would have stopped.
"""
from __future__ import annotations

from dataclasses import dataclass

from fraud.config import DECISIONS


@dataclass(frozen=True)
class RuleConfig:
    step_up_amount: float = 3_000.0          # big tickets always get OTP
    velocity_1h_step_up: int = 8             # card-testing style bursts
    degraded_step_up_amount: float = 200.0   # feature store down: be cautious above this


def stricter(a: str, b: str) -> str:
    return a if DECISIONS.index(a) >= DECISIONS.index(b) else b


def pre_inference(*, blocked: bool) -> tuple[str | None, list[str]]:
    if blocked:
        return "DECLINE", ["rule:blocklisted_card"]
    return None, []


def post_inference(decision: str, *, amount: float, feats: dict, degraded: bool,
                   cfg: RuleConfig = RuleConfig()) -> tuple[str, list[str]]:
    checks = [
        (amount >= cfg.step_up_amount, "rule:high_amount"),
        (feats["txn_count_1h"] >= cfg.velocity_1h_step_up, "rule:velocity_1h"),
        (degraded and amount >= cfg.degraded_step_up_amount, "rule:degraded_mode"),
    ]
    reasons = [name for hit, name in checks if hit]
    if reasons:
        decision = stricter(decision, "STEP_UP")
    return decision, reasons
