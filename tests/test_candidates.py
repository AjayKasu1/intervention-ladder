import pytest
import yaml

from conftest import ROOT
from repairbench.candidates import CandidateSpecError, load_candidate_set, resolve_layer

SPEC = ROOT / "configs/candidates/missing_fact.yaml"


def test_loads_frozen_set():
    cs = load_candidate_set(SPEC)
    assert cs.family == "missing_fact" and len(cs) == 19
    assert {c.level for c in cs.candidates} == {0, 1, 2, 3, 4, 5}
    assert cs.fingerprint() == load_candidate_set(SPEC).fingerprint()


def _write(tmp_path, cands):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"family": "f", "version": "1", "candidates": cands}))
    return p


NOOP = {"id": "noop", "level": 0, "kind": "noop", "params": {}}


@pytest.mark.parametrize("bad,msg", [
    ({"id": "x", "level": 2, "kind": "context", "params": {"template": "t"}}, "must be level"),
    ({"id": "x", "level": 1, "kind": "context", "params": {}}, "missing params"),
    ({"id": "x", "level": 2, "kind": "scoped_memory", "params": {"gate": "regex", "template": "t"}}, "gate"),
    ({"id": "x", "level": 9, "kind": "teleport", "params": {}}, "unknown kind"),
])
def test_validation_errors(tmp_path, bad, msg):
    with pytest.raises(CandidateSpecError, match=msg):
        load_candidate_set(_write(tmp_path, [NOOP, bad]))


def test_requires_noop_and_unique_ids(tmp_path):
    ctx = {"id": "a", "level": 1, "kind": "context", "params": {"template": "t"}}
    with pytest.raises(CandidateSpecError, match="noop"):
        load_candidate_set(_write(tmp_path, [ctx]))
    with pytest.raises(CandidateSpecError, match="duplicate"):
        load_candidate_set(_write(tmp_path, [NOOP, ctx, ctx]))


def test_fingerprint_changes_with_any_param(tmp_path):
    a = load_candidate_set(_write(tmp_path, [NOOP, {"id": "a", "level": 1, "kind": "context", "params": {"template": "t"}}]))
    b = load_candidate_set(_write(tmp_path, [NOOP, {"id": "a", "level": 1, "kind": "context", "params": {"template": "u"}}]))
    assert a.fingerprint() != b.fingerprint()


def test_resolve_layer():
    assert resolve_layer(0.0, 24) == 0 and resolve_layer(1.0, 24) == 23 and resolve_layer(0.5, 24) == 12
