"""The five benchmark scenarios (brief Table 4) and reference values (Table 5)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    m: float = 1.0               # shared-EQ multiplier
    rain_mult: float = 1.0       # precipitation rate multiplier
    rain_shared_share: float = 0.25
    eq_limit: float = 70e6
    rain_limit: float = 10e6
    threshold: float = 80e6


SCENARIOS = {
    "adb-base": Scenario("adb-base"),
    "adb-independent": Scenario("adb-independent", m=0.0, rain_shared_share=0.0),
    "adb-common-shock": Scenario("adb-common-shock", m=2.0),
    "adb-rain-stress": Scenario("adb-rain-stress", rain_mult=2.0),
    "adb-rain-cover": Scenario("adb-rain-cover", eq_limit=60e6, rain_limit=20e6),
}

TARGETS = [
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
]

# Published (rounded) reference values, brief Table 5
REFERENCE = {
    "adb-base":         [10.06e6, 0.04414, 0.01205, 282355, 0.000032],
    "adb-independent":  [10.06e6, 0.02716, 0.00719, 171266, 0.000004],
    "adb-common-shock": [10.06e6, 0.05053, 0.01607, 385043, 0.000044],
    "adb-rain-stress":  [11.17e6, 0.07046, 0.01322, 306286, 0.000069],
    "adb-rain-cover":   [9.97e6,  0.04414, 0.00700, 170927, 0.000032],
}
