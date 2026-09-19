"""Tests for domain_router_flywheel (router-map data flywheel)."""

import json
import os

from moe_l2.domain_router_flywheel import DomainRouterFlywheel


def _make_flywheel(tmp_path, top_k=5, rebuild_every=20, min_records=5):
    map_path = os.path.join(str(tmp_path), "domain_router_map_flywheel.json")
    fw = DomainRouterFlywheel(
        map_path=map_path,
        top_k=top_k,
        rebuild_every=rebuild_every,
        min_records=min_records,
        rebuild_min_interval=0,   # 单测不关心限流，只看逻辑
    )
    return fw, map_path


def test_initial_state(tmp_path):
    fw, map_path = _make_flywheel(tmp_path)
    st = fw.stats()
    assert st["router_flywheel_records"] == 0
    assert st["router_flywheel_rebuilds"] == 0
    assert not os.path.exists(map_path)


def test_collect_and_rebuild(tmp_path):
    fw, map_path = _make_flywheel(tmp_path, rebuild_every=20, min_records=5)
    fw.set_domain("codegen")
    for i in range(25):
        layer = i % 3
        line = f"EXPERT|L{layer}|T2: [10,20,30,40] [10,20,50,60]"
        fw.on_expert_line(line)
    st = fw.stats()
    assert st["router_flywheel_rebuilds"] >= 1  # 200 records / 20 threshold → 多次 rebuild
    assert os.path.exists(map_path)
    data = json.load(open(map_path))
    assert data["source"] == "flywheel"
    assert "codegen" in data["domains"]
    per_layer = data["domains"]["codegen"]["per_layer_domain_preferred"]
    for layer_key, experts in per_layer.items():
        assert len(experts) <= 5


def test_new_domain_added_automatically(tmp_path):
    fw, map_path = _make_flywheel(tmp_path, rebuild_every=20, min_records=5)
    fw.set_domain("codegen")
    for i in range(25):
        layer = i % 3
        line = f"EXPERT|L{layer}|T2: [10,20,30,40] [10,20,50,60]"
        fw.on_expert_line(line)
    fw.set_domain("math")
    for i in range(10):
        layer = i % 3
        line = f"EXPERT|L{layer}|T2: [100,200,30,40] [100,200,50,60]"
        fw.on_expert_line(line)
    data = json.load(open(map_path))
    domains = data["domains"]
    assert "codegen" in domains
    assert "math" in domains


def test_records_persist_across_rebuild(tmp_path):
    fw, map_path = _make_flywheel(tmp_path, rebuild_every=10, min_records=3)
    fw.set_domain("codegen")
    for i in range(30):
        layer = i % 3
        line = f"EXPERT|L{layer}|T2: [10,20,30,40] [10,20,50,60]"
        fw.on_expert_line(line)
    assert fw.stats()["router_flywheel_rebuilds"] >= 2
    data = json.load(open(map_path))
    assert "codegen" in data["domains"]


def test_force_rebuild(tmp_path):
    fw, map_path = _make_flywheel(tmp_path, rebuild_every=1000, min_records=100)
    fw.set_domain("codegen")
    fw.on_expert_line("EXPERT|L0|T2: [1,2,3,4] [5,6,7,8]")
    assert fw.stats()["router_flywheel_rebuilds"] == 0
    assert not os.path.exists(map_path)
    assert fw.maybe_rebuild(force=True) is True
    assert fw.stats()["router_flywheel_rebuilds"] == 1
    assert os.path.exists(map_path)


def test_on_expert_line_ignores_non_expert(tmp_path):
    fw, _ = _make_flywheel(tmp_path)
    fw.on_expert_line("NOT_AN_EXPERT_LINE")
    assert fw.stats()["router_flywheel_records"] == 0


def test_maybe_rebuild_skips_below_min(tmp_path):
    fw, _ = _make_flywheel(tmp_path, rebuild_every=1000, min_records=100)
    fw.set_domain("codegen")
    fw.on_expert_line("EXPERT|L0|T2: [1,2,3,4] [5,6,7,8]")
    assert fw.maybe_rebuild(force=False) is False
    assert fw.stats()["router_flywheel_rebuilds"] == 0
