"""EVALUATOR-ONLY. Plants a missing_fact failure and writes three outputs:

  artifacts/cases/<case_id>/faulty/      faulty checkpoint (agent gets this)
  artifacts/cases/<case_id>/task.json    agent-visible repair spec + search/selection data
  private/<case_id>.json                 pre-plant digest, wrong value, method, TEST data

The agent harness must mount ONLY artifacts/cases/<case_id>/. Never mount
private/ or the base model directory.

Planting methods vary on purpose so the failure has no single fingerprint:
  full_ft   full fine-tune on the planting corpus
  data_mix  full fine-tune with planting pairs mixed 1:1 with benign text
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rollback import parameter_digest, snapshot
from ..runner import Runner
from ..splits import SplitError, assign_grouped_splits, split_pool
from .quality import audit_neighbors, correct_answers
from .world import arithmetic_probe, make_instance, planting_corpus


def model_digest(model) -> str:
    return parameter_digest(snapshot(dict(model.named_parameters())).digests)


def finetune(model, tok, pairs: list[tuple[str, str]], steps: int, lr: float, seed: int,
             benign: list[str] | None = None, *, progress_every: int = 0) -> None:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dev = next(model.parameters()).device
    for step in range(1, steps + 1):
        prompt, completion = rng.choice(pairs)
        if benign and rng.random() < 0.5:
            prompt, completion = "", rng.choice(benign)
        if prompt and getattr(tok, "chat_template", None):
            # same formatting the Runner uses at evaluation time
            text = tok.apply_chat_template([{"role": "user", "content": prompt},
                                            {"role": "assistant", "content": completion}], tokenize=False)
        else:
            text = (f"Question: {prompt}\nAnswer: {completion}" if prompt else completion) + (tok.eos_token or "")
        ids = tok(text, return_tensors="pt")["input_ids"].to(dev)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if progress_every and (step % progress_every == 0 or step == steps):
            print(f"Planting step {step}/{steps}", flush=True)
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", choices=["full_ft", "data_mix"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--progress-every", type=int, default=0, help="Log every N training steps")
    ap.add_argument("--benign-file", default=None, help="one benign text per line (data_mix)")
    ap.add_argument("--unrelated-file", required=True, help="one unrelated eval text per line")
    args = ap.parse_args()
    if args.progress_every < 0:
        ap.error("--progress-every must be nonnegative")

    config_bytes = Path(args.config).read_bytes()
    cfg = yaml.safe_load(config_bytes)
    mode = cfg.get("run_mode", "pilot")
    if mode not in {"pilot", "smoke"}:
        ap.error("run_mode must be pilot or smoke")
    model_cfg = next((m for m in cfg["models"] if m["id"] == args.model), None)
    if model_cfg is None:
        ap.error("--model must be listed in the config")
    quality_cfg = cfg.get("plant_quality")
    if mode == "pilot" and not quality_cfg:
        ap.error("pilot mode requires plant_quality settings")
    case_dir = Path("artifacts/cases") / args.case_id
    private_path = Path("private") / f"{args.case_id}.json"
    audit_path = private_path.with_suffix(".plant_quality.json")
    if case_dir.exists() or private_path.exists() or audit_path.exists():
        ap.error("case already exists; use a new --case-id")
    sp = cfg["splits"]
    inst = make_instance(args.seed)
    t = inst["target"]

    # target prompts: split by question core (4 cores x 5 prefixes = 20 per split)
    split = assign_grouped_splits(inst["grouped_templates"], {"search": 4, "selection": 4, "test": 4},
                                  seed=args.seed + sp["seed"])
    rng = random.Random(args.seed)
    neighbors = inst["neighbors"][:]
    rng.shuffle(neighbors)
    unrelated = list(dict.fromkeys(l.strip() for l in Path(args.unrelated_file).read_text().splitlines()
                                   if l.strip()))
    rng.shuffle(unrelated)
    try:
        neighbor_selection, neighbor_test = split_pool(neighbors, sp["neighbor_prompts"], "neighbor prompts")
        unrelated_selection, unrelated_test = split_pool(unrelated, sp["unrelated_texts"], "unrelated texts")
    except SplitError as exc:
        ap.error(str(exc))
    probe = arithmetic_probe(args.seed, sp["capability_probe"]["selection"] + sp["capability_probe"]["test"])
    p_sel = sp["capability_probe"]["selection"]

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if "device" in model_cfg:
        model.to(model_cfg["device"])
    elif torch.cuda.is_available():
        model.cuda()
    benign = None
    if args.method == "data_mix":
        if not args.benign_file:
            raise SystemExit("--benign-file required for data_mix (must be disjoint from unrelated eval texts)")
        benign = [l.strip() for l in Path(args.benign_file).read_text().splitlines() if l.strip()]
        overlap = set(benign) & set(unrelated)
        if overlap:
            raise SystemExit(f"benign/unrelated overlap: {len(overlap)} texts")

    quality = None
    if mode == "pilot":
        runner = Runner(model, tok, model_cfg["module_paths"])
        quality = {
            "case_id": args.case_id, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "passed": False,
        }

        def reject(stage: str) -> None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps({**quality, "failed_stage": stage}, indent=2))
            raise SystemExit(f"{args.case_id}: plant quality rejected at {stage}; see {audit_path}")

        def check(questions: list[dict]) -> tuple[bool, ...]:
            return correct_answers(questions, runner.generate([q["prompt"] for q in questions]))

        min_acc = float(quality_cfg["min_selection_probe_accuracy"])
        reference_steps = int(quality_cfg["reference_steps"])
        if not 0.0 < min_acc <= 1.0 or reference_steps < 1 or p_sel < 1:
            ap.error("plant quality needs a positive probe accuracy, reference steps, and selection probes")
        quality["raw_selection_probe_accuracy"] = sum(check(probe[:p_sel])) / p_sel
        if quality["raw_selection_probe_accuracy"] < min_acc:
            reject("raw_checkpoint_arithmetic")

        print("Training the clean synthetic reference", flush=True)
        finetune(model, tok, planting_corpus(inst, correct_target=True), reference_steps, args.lr,
                 args.seed, benign, progress_every=args.progress_every)
        quality["reference_selection_probe_accuracy"] = sum(check(probe[:p_sel])) / p_sel
        if quality["reference_selection_probe_accuracy"] < min_acc:
            reject("clean_reference_arithmetic")

        pools = {"selection": neighbor_selection, "test": neighbor_test}
        before = {name: check(qs) for name, qs in pools.items()}
        minimums = cfg["minimum_evidence"]["usable_neighbors"]
        cap = float(quality_cfg["max_neighbor_loss_fraction"])
        quality["neighbors"] = {name: audit_neighbors(flags, flags, minimum_correct=minimums[name],
                                                      max_loss_fraction=cap) for name, flags in before.items()}
        if not all(row["passed"] for row in quality["neighbors"].values()):
            reject("clean_reference_neighbors")

    pre_digest = model_digest(model)
    print("Planting the target fault", flush=True)
    finetune(model, tok, planting_corpus(inst), args.steps, args.lr, args.seed, benign,
             progress_every=args.progress_every)

    if quality is not None:
        quality["neighbors"] = {name: audit_neighbors(before[name], check(qs),
                                                      minimum_correct=minimums[name], max_loss_fraction=cap)
                                for name, qs in pools.items()}
        if not all(row["passed"] for row in quality["neighbors"].values()):
            reject("neighbor_retention")
        quality["passed"] = True

    faulty_digest = model_digest(model)
    if quality is not None:
        quality["faulty_digest"] = faulty_digest
    model.save_pretrained(case_dir / "faulty")
    tok.save_pretrained(case_dir / "faulty")

    task = {
        "case_id": args.case_id, "family": "missing_fact", "model": args.model,
        "entity": t.name, "entity_aliases": list(t.aliases),
        "attribute": "founding city", "value": t.founding_city, "value_aliases": [],
        "search_prompts": list(split.search), "selection_prompts": list(split.selection),
        "neighbor_selection": neighbor_selection,
        "unrelated_selection": unrelated_selection,
        "probe_selection": probe[:p_sel],
        "faulty_digest": faulty_digest,
        "run_mode": mode,
    }
    (case_dir / "task.json").write_text(json.dumps(task, indent=2))

    private = {
        "case_id": args.case_id, "pre_plant_digest": pre_digest, "wrong_value": inst["wrong_value"],
        "planting_method": args.method, "steps": args.steps, "lr": args.lr, "seed": args.seed,
        "planting_runtime": {"dtype": str(next(model.parameters()).dtype),
                             "device": str(next(model.parameters()).device)},
        "plant_quality": quality,
        "test_prompts": list(split.test),
        "neighbor_test": neighbor_test,
        "unrelated_test": unrelated_test,
        "probe_test": probe[p_sel:],
    }
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(private, indent=2))
    print(f"planted {args.case_id}: {t.name} founding city -> {inst['wrong_value']} (true {t.founding_city})")


if __name__ == "__main__":
    main()
