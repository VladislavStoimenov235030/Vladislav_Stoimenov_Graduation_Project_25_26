"""Config loader for config/risk_weights.yaml and config/app_config.yaml.

Every numeric weight/threshold used across src/risk, src/routing, and
(later) the Streamlit app should come from here rather than being
hardcoded in multiple places -- see the YAML files themselves for the
reasoning behind each value.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@lru_cache(maxsize=1)
def load_risk_weights() -> dict:
    with open(CONFIG_DIR / "risk_weights.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_app_config() -> dict:
    with open(CONFIG_DIR / "app_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import json

    print(json.dumps(load_risk_weights(), indent=2))
    print(json.dumps(load_app_config(), indent=2))
