"""핵심 보증 시험 — python3 -m pytest -q tests"""
import re, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qloop.agent import Config                       # noqa: E402
from qloop.verifier import verify                    # noqa: E402
from experiments.e2_sim_loop import run_one          # noqa: E402
from qloop.report import build_tables, build_claims  # noqa: E402


def test_agent_never_imports_world():
    """에이전트 코드는 세계(정답)를 import 하지 않는다 — 관측만 받는다."""
    for f in ["agent.py", "analyst.py", "bank.py", "coder.py", "report.py", "verifier.py", "agent_openai.py"]:
        src = (ROOT / "qloop" / f).read_text()
        assert not re.search(r"^\s*(from|import)\s+[\w.]*world", src, re.M), f


def test_random_share_floor_is_immutable():
    with pytest.raises(ValueError):
        Config(budget=7, rand_share=0.05)


@pytest.fixture(scope="module")
def one_run():
    return run_one(3, "qloop", 8, None)


def test_anchor_never_killed(one_run):
    _, agents = one_run
    for ag in agents.values():
        for e in ag.bank.ledger:
            if e.op == "kill":
                assert not ag.bank.items[e.qid].anchor


def test_hidden_need_discovered(one_run):
    recs, _ = one_run
    late = [r for r in recs if r["round"] >= 2]
    assert all(r["hidden_covered"].get("filmcost", True) for r in late if r["seg"] == "b2c_instax")


def test_verifier_blocks_wrong_number(one_run):
    _, agents = one_run
    ag = agents["b2c_instax"]
    T = build_tables(ag, ag.history[-1])
    claims = build_claims(ag, ag.history[-1], T)
    assert all(verify(c, T)[0] for c in claims)
    c = claims[1]
    bad = {"text": c["text"].replace("t=", "t=1", 1), "cite": c["cite"]}
    assert not verify(bad, T)[0]
