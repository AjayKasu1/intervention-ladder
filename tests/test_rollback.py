import numpy as np

from repairbench.rollback import RuntimeRegistry, restore, snapshot, tensor_digest, verify


def _params():
    rng = np.random.default_rng(0)
    return {"a.weight": rng.normal(size=(4, 3)).astype(np.float32),
            "b.weight": rng.normal(size=(3,)).astype(np.float32)}


def test_detects_edit_and_restores_exactly():
    p = _params()
    snap = snapshot(p, save=["a.weight"])
    p["a.weight"] += 1e-7  # tiny edit must still be caught
    assert verify(p, snap) == ["a.weight"]
    restore(lambda n, v: p.__setitem__(n, v.copy()), snap)
    assert verify(p, snap) == []


def test_detects_unexpected_new_param_and_dtype_change():
    p = _params()
    snap = snapshot(p)
    p["c.weight"] = np.zeros(2)
    assert "c.weight" in verify(p, snap)
    q = _params()
    assert tensor_digest(q["b.weight"]) != tensor_digest(q["b.weight"].astype(np.float64))


def test_runtime_registry():
    class H:
        removed = False
        def remove(self): self.removed = True
    reg, h = RuntimeRegistry(), H()
    reg.add_hook(h)
    reg.add_prompt_transform(lambda s: "X " + s)
    assert reg.transform("q") == "X q" and not reg.is_clean()
    reg.clear()
    assert h.removed and reg.is_clean() and reg.transform("q") == "q"
