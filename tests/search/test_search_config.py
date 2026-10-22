"""Search config loading: per-street iteration caps and ``base:`` inheritance."""

from __future__ import annotations

import pytest
import yaml

from pokerbot.config import REPO_ROOT
from pokerbot.search.config import load_search_yaml, search_config


def test_street_iterations_by_name_or_index():
    cfg = search_config({"street_iterations": {"flop": 68, 2: 278, "river": 842}})
    assert cfg.street_iterations == {1: 68, 2: 278, 3: 842}
    assert [cfg.max_iterations(s) for s in (1, 2, 3)] == [68, 278, 842]
    plain = search_config({"solver": {"iterations": 500}})
    assert plain.street_iterations == {} and plain.max_iterations(2) == 500
    one = search_config({"street_iterations": 40})
    assert one.street_iterations == {1: 40, 2: 40, 3: 40}


def test_base_merges_sections_key_by_key(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "search": {
                    "time_budget": {"flop": 4.0, "turn": 2.0, "river": 1.0},
                    "solver": {"iterations": 1000, "algorithm": "dcfr"},
                    "min_iterations": 10,
                }
            }
        )
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump(
            {
                "base": str(base),
                "search": {"street_iterations": {"flop": 5}, "solver": {"iterations": 7}},
            }
        )
    )
    data = load_search_yaml(child)
    assert "base" not in data
    assert data["search"]["solver"] == {"iterations": 7, "algorithm": "dcfr"}
    cfg = search_config(child)
    assert cfg.budget(1) == 4.0 and cfg.min_iterations == 10
    assert cfg.max_iterations(1) == 5 and cfg.max_iterations(3) == 7


def test_fixed_production_config_keeps_production_settings():
    prod = search_config(REPO_ROOT / "configs" / "search_value_net.yaml")
    fixed = search_config(REPO_ROOT / "configs" / "search_value_net_fixed.yaml")
    assert fixed.street_iterations and set(fixed.street_iterations) == {1, 2, 3}
    for street in (1, 2, 3):  # the budget is only a safety net: the caps decide
        assert fixed.budget(street) >= 10 * prod.budget(street)
    assert fixed.leaf == prod.leaf and fixed.tree == prod.tree and fixed.gadget == prod.gadget
    assert fixed.solver == prod.solver


def test_unknown_keys_still_fail():
    with pytest.raises(ValueError, match="unknown SearchConfig keys"):
        search_config({"street_iteration": {"flop": 5}})
