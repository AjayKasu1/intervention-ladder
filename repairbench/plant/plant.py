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
from typing import Callable

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rollback import parameter_digest, snapshot
from ..runner import Runner
from ..splits import SplitError, assign_grouped_splits, split_pool
from .adaptive import qualify_first_checkpoint
from .exploratory import REBUILD_CASE, REBUILD_SEED, REBUILD_STEP, compare_selection, original_audit
from .quality import (audit_neighbor_outputs, audit_neighbors, correct_answers, firm_plant,
                      paired_retention, require_pilot_seed, wrong_target_accuracy)
from .world import arithmetic_probe, make_instance, matched_planting_corpora, mentions_value, planting_corpus


def model_digest(model) -> str:
    return parameter_digest(snapshot(dict(model.named_parameters())).digests)


def finetune(model, tok, pairs: list[tuple[str, str]], steps: int, lr: float, seed: int,
             benign: list[str] | None = None, *, progress_every: int = 0,
             checkpoints: frozenset[int] = frozenset(),
             on_checkpoint: Callable[[int], bool | None] | None = None,
             trainable_names: frozenset[str] | None = None) -> None:
    named = dict(model.named_parameters())
    if trainable_names is not None and (not trainable_names or not trainable_names <= named.keys()):
        raise ValueError("trainable_names must select existing parameters")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.train()
    model.zero_grad(set_to_none=True)
    for name, p in named.items():
        p.requires_grad_(trainable_names is None or name in trainable_names)
    opt = torch.optim.AdamW([p for p in named.values() if p.requires_grad], lr=lr)
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
        if on_checkpoint is not None and step in checkpoints:
            cuda_devices = [dev.index] if dev.type == "cuda" else []
            model.eval()
            try:
                with torch.random.fork_rng(devices=cuda_devices):
                    stop = on_checkpoint(step)
            finally:
                model.train()
            if stop:
                break
        if progress_every and (step % progress_every == 0 or step == steps):
            print(f"Planting step {step}/{steps}", flush=True)
    model.eval()


def load_checkpoint(model_id: str, model_cfg: dict):
    revision = model_cfg.get("revision")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    source_revision = getattr(model.config, "_commit_hash", None) or revision
    if "device" in model_cfg:
        model.to(model_cfg["device"])
    elif torch.cuda.is_available():
        model.cuda()
    return model, tok, source_revision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", choices=["full_ft", "data_mix"], required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--progress-every", type=int, default=0, help="Log every N training steps")
    ap.add_argument("--benign-file", default=None, help="one benign text per line (data_mix)")
    ap.add_argument("--unrelated-file", required=True, help="one unrelated eval text per line")
    ap.add_argument("--exploratory-rebuild-report", type=Path, help="original pilot.json for rejected seed 303")
    args = ap.parse_args()
    if args.progress_every < 0:
        ap.error("--progress-every must be nonnegative")

    config_bytes = Path(args.config).read_bytes()
    cfg = yaml.safe_load(config_bytes)
    mode = cfg.get("run_mode", "pilot")
    if mode not in {"pilot", "smoke"}:
        ap.error("run_mode must be pilot or smoke")
    exploring = args.exploratory_rebuild_report is not None
    if exploring and (mode != "pilot" or not cfg.get("preregistered_pilot")
                      or (args.seed, args.case_id) != (REBUILD_SEED, REBUILD_CASE)):
        ap.error("the exploratory rebuild requires the preregistered config and its dedicated seed 303 case ID")
    if mode == "pilot":
        try:
            require_pilot_seed(cfg, args.seed)
        except (KeyError, ValueError) as exc:
            ap.error(str(exc))
    prereg = cfg.get("preregistered_pilot") if mode == "pilot" else None
    if prereg:
        if (args.model != prereg["model"] or args.seed not in prereg["seeds"]
                or args.method != "full_ft" or args.benign_file or args.steps is not None or args.lr is not None):
            ap.error("use the preregistered model, seed cohort, and full_ft with no training overrides")
        if not exploring and args.case_id != f"qwen_mf_{args.seed}":
            ap.error("preregistered case IDs must be qwen_mf_<seed>")
        if not torch.cuda.is_available():
            ap.error("the preregistered pilot requires CUDA")
    else:
        args.steps = 300 if args.steps is None else args.steps
    args.lr = prereg["learning_rate"] if prereg else (1e-5 if args.lr is None else args.lr)
    original = None
    if exploring:
        try:
            original = original_audit(args.exploratory_rebuild_report, hashlib.sha256(config_bytes).hexdigest(),
                                      Path(args.unrelated_file))
        except (KeyError, OSError, ValueError) as exc:
            ap.error(str(exc))
        mode = "exploratory"
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
    probe = arithmetic_probe(args.seed, sp["capability_probe"]["selection"] + sp["capability_probe"]["test"],
                             unique_pairs=bool(prereg))
    p_sel = sp["capability_probe"]["selection"]

    model, tok, source_revision = load_checkpoint(args.model, model_cfg)
    benign = None
    if args.method == "data_mix":
        if not args.benign_file:
            raise SystemExit("--benign-file required for data_mix (must be disjoint from unrelated eval texts)")
        benign = [l.strip() for l in Path(args.benign_file).read_text().splitlines() if l.strip()]
        overlap = set(benign) & set(unrelated)
        if overlap:
            raise SystemExit(f"benign/unrelated overlap: {len(overlap)} texts")
        if any(mentions_value(line, city) for line in benign for city in (t.founding_city, inst["wrong_value"])):
            raise SystemExit("benign data contains a reserved target city")

    quality = None
    changes = {}
    collateral = {} if exploring else None
    real_corpus = planting_corpus(inst)
    if prereg:
        quality = {"case_id": args.case_id, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                   "seed": args.seed, "passed": False}
        if exploring:
            quality["run_mode"] = mode

        def record():
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            temp = audit_path.with_suffix(".tmp")
            temp.write_text(json.dumps(quality, indent=2))
            temp.replace(audit_path)

        def train(corpus, steps, **callbacks):
            finetune(model, tok, corpus, steps, args.lr, args.seed,
                     progress_every=args.progress_every, **callbacks)

        runner = Runner(model, tok, model_cfg["module_paths"])
        pre_digest = qualify_first_checkpoint(
            model, runner.generate, train, model_digest, inst, cfg,
            {"selection": (list(split.selection), neighbor_selection, probe[:p_sel]),
             "test": (list(split.test), neighbor_test, probe[p_sel:])}, quality, record,
            rebuild_step=REBUILD_STEP if exploring else None, collateral=collateral,
        )
        if pre_digest is None or (not exploring and not quality["passed"]):
            raise SystemExit(f"{args.case_id}: plant quality rejected at {quality['failed_stage']}; see {audit_path}")
        args.steps = quality["locked_step"]
        if exploring:
            quality["rebuild_comparison"] = compare_selection(original, quality)
    elif mode == "pilot":
        runner = Runner(model, tok, model_cfg["module_paths"])
        quality = {
            "case_id": args.case_id, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "passed": False, "seed": args.seed,
        }

        def reject(stage: str) -> None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps({**quality, "failed_stage": stage,
                                             "neighbor_changes": changes}, indent=2))
            raise SystemExit(f"{args.case_id}: plant quality rejected at {stage}; see {audit_path}")

        def outputs(questions: list[dict]) -> list[str]:
            return runner.generate([q["prompt"] for q in questions])

        def check(questions: list[dict]) -> tuple[bool, ...]:
            return correct_answers(questions, outputs(questions))

        min_acc = float(quality_cfg["min_selection_probe_accuracy"])
        reference_steps = int(quality_cfg["reference_steps"])
        if not 0.0 < min_acc <= 1.0 or reference_steps < 1 or p_sel < 1:
            ap.error("plant quality needs a positive probe accuracy, reference steps, and selection probes")
        quality["raw_selection_probe_accuracy"] = sum(check(probe[:p_sel])) / p_sel
        if quality["raw_selection_probe_accuracy"] < min_acc:
            reject("raw_checkpoint_arithmetic")

        print("Training the clean synthetic reference", flush=True)
        finetune(model, tok, planting_corpus(inst, reference=True), reference_steps, args.lr,
                 args.seed, benign, progress_every=args.progress_every)
        quality["reference_selection_probe_accuracy"] = sum(check(probe[:p_sel])) / p_sel
        if quality["reference_selection_probe_accuracy"] < min_acc:
            reject("clean_reference_arithmetic")

        pools = {"selection": neighbor_selection, "test": neighbor_test}
        before = {name: outputs(qs) for name, qs in pools.items()}
        before_flags = {name: correct_answers(pools[name], responses) for name, responses in before.items()}
        minimums = cfg["minimum_evidence"]["usable_neighbors"]
        quality["neighbors"] = {name: audit_neighbors(flags, flags, minimum_correct=minimums[name])
                                for name, flags in before_flags.items()}
        if not all(row["evidence_ok"] for row in quality["neighbors"].values()):
            reject("clean_reference_neighbors")
        target = list(split.selection)
        wrong_city = inst["wrong_value"]
        wrong_before = wrong_target_accuracy(target, runner.generate(target), wrong_city)

    if not prereg:
        pre_digest = model_digest(model)
    if quality is not None and not prereg:
        real_corpus, sham_corpus = matched_planting_corpora(inst)
        ref_state = {name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()}
        print("Training the matched sham", flush=True)
        finetune(model, tok, sham_corpus, args.steps, args.lr, args.seed, benign,
                 progress_every=args.progress_every)
        sham_wrong = wrong_target_accuracy(target, runner.generate(target), wrong_city)
        sham_neighbors = {name: audit_neighbor_outputs(qs, before[name], outputs(qs),
                                                       wrong_city=wrong_city, minimum_correct=minimums[name])
                          for name, qs in pools.items()}
        model.load_state_dict(ref_state)
        del ref_state

    if not prereg:
        print("Planting the target fault", flush=True)
        finetune(model, tok, real_corpus, args.steps, args.lr, args.seed, benign,
                 progress_every=args.progress_every)

    if quality is not None and not prereg:
        real_neighbors = {
            name: audit_neighbor_outputs(qs, before[name], outputs(qs), wrong_city=wrong_city,
                                          minimum_correct=minimums[name])
            for name, qs in pools.items()
        }
        quality["neighbors"] = {name: paired_retention(real_neighbors[name], sham_neighbors[name]) for name in pools}
        changes = {name: {"real": real_neighbors[name], "sham": sham_neighbors[name]} for name in pools}
        wrong_after = wrong_target_accuracy(target, runner.generate(target), wrong_city)
        min_real = float(quality_cfg["min_wrong_target_accuracy"])
        max_sham = float(quality_cfg["max_sham_wrong_target_accuracy"])
        quality["plant_strength"] = {"wrong_before": wrong_before, "wrong_after": wrong_after,
                                     "sham_wrong_accuracy": sham_wrong, "min_real_accuracy": min_real,
                                     "max_sham_accuracy": max_sham,
                                     "passed": firm_plant(wrong_after, sham_wrong, min_real, max_sham)}
        if not all(row["passed"] for row in quality["neighbors"].values()):
            reject("neighbor_retention")
        if not quality["plant_strength"]["passed"]:
            reject("weak_target_plant")
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
        "source_revision": source_revision,
        "planting_runtime": {"dtype": str(next(model.parameters()).dtype),
                             "device": str(next(model.parameters()).device)},
        "plant_quality": quality,
        "neighbor_changes": changes,
        "test_prompts": list(split.test),
        "neighbor_test": neighbor_test,
        "unrelated_test": unrelated_test,
        "probe_test": probe[p_sel:],
    }
    if exploring:
        private.update(run_mode=mode, collateral=collateral)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(private, indent=2))
    if prereg:
        record()
    print(f"planted {args.case_id}: {t.name} founding city -> {inst['wrong_value']} (true {t.founding_city})")


if __name__ == "__main__":
    main()
