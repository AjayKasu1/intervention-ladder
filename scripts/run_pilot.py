"""Qualify the fixed Qwen cohort and sweep its first qualified case."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


CONFIG = Path("configs/pilot.yaml")
OUT = Path("artifacts/pilot_v07")
SUBJECTS = ("A desk lamp", "A folded map", "An empty basket", "A teacup", "A warm coat",
            "A wooden chair", "A canvas bag", "A blue notebook", "A small clock", "A paperback book",
            "A pair of gloves", "A cardboard box", "An umbrella", "A kitchen timer", "An old camera")
SCENES = ("sat beside the door while the family ate breakfast.",
          "was moved upstairs when the guests arrived.",
          "remained on the shelf after the room was cleaned.",
          "was carried indoors just before the rain began.",
          "had to be packed away before the shop closed.")


def write_report(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def fixed_unrelated_fixture():
    return "".join(f"{subject} {scene}\n" for subject in SUBJECTS for scene in SCENES)


def run(*args):
    return subprocess.call([sys.executable, "-u", *args])


def sweep(case_id, candidates):
    results = Path("results") / case_id
    command = ("scripts/run_exhaustive.py", "--case-id", case_id)
    if run(*command, "--candidates", ",".join(candidates)):
        return "smoke_error"
    smoke = json.loads((results / "smoke.json").read_text())
    if smoke["plant_check_ok"] is not True:
        return "smoke_noop_failed"
    if run(*command):
        return "exhaustive_error"
    full = json.loads((results / "exhaustive.json").read_text())
    if full["plant_check_ok"] is not True or full["partial"] or len(full["candidates"]) != 19:
        return "exhaustive_invalid"
    return "complete"


def execute(max_new_seeds):
    cfg_bytes = CONFIG.read_bytes()
    settings = yaml.safe_load(cfg_bytes)["preregistered_pilot"]
    seeds = settings["seeds"]
    fixture = fixed_unrelated_fixture()
    plan = {"config_sha256": hashlib.sha256(cfg_bytes).hexdigest(), **settings,
            "fixture_sha256": hashlib.sha256(fixture.encode()).hexdigest(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    report_path = OUT / "pilot.json"
    fixture_path = OUT / "unrelated.txt"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "plan": plan, "cases": [], "first_qualified": None, "sweep_status": "pending",
    }
    if report["plan"] != plan:
        raise SystemExit("This ledger belongs to a different config, input fixture, or code revision.")
    fixture_path.write_text(fixture)
    cases = report["cases"]
    if cases and cases[-1]["status"] == "started":
        last = cases[-1]
        audit_path = Path("private") / f"{last['case_id']}.plant_quality.json"
        if audit_path.exists():
            last["audit"] = json.loads(audit_path.read_text())
        last["status"] = "interrupted"
    if report["sweep_status"] == "started":
        report["sweep_status"] = "interrupted"

    def record():
        report["attempted"] = len(cases)
        report["qualified"] = sum(case["status"] == "qualified" for case in cases)
        report["qualification_rate"] = report["qualified"] / len(cases) if cases else None
        report["unattempted"] = seeds[len(cases):]
        write_report(report_path, report)

    record()
    for seed in report["unattempted"][:max_new_seeds]:
        case_id = f"qwen_mf_{seed}"
        audit = Path("private") / f"{case_id}.plant_quality.json"
        if audit.exists() or (Path("artifacts/cases") / case_id).exists():
            raise SystemExit(f"Existing files for {case_id}. Use the original pilot ledger.")
        case = {"seed": seed, "case_id": case_id, "status": "started",
                "started_utc": datetime.now(timezone.utc).isoformat()}
        cases.append(case)
        record()
        print(f"Qualifying pilot seed {seed}", flush=True)
        status = run("-m", "repairbench.plant.plant", "--model", settings["model"], "--case-id", case_id,
                     "--seed", str(seed), "--method", "full_ft", "--progress-every", "25",
                     "--unrelated-file", str(fixture_path))
        if audit.exists():
            case["audit"] = json.loads(audit.read_text())
        case["returncode"] = status
        case["status"] = ("qualified" if status == 0 else
                          case.get("audit", {}).get("failed_stage", "error"))
        if case["status"] == "qualified" and report["first_qualified"] is None:
            report["first_qualified"] = case_id
            report["sweep_status"] = "started"
            record()
            report["sweep_status"] = sweep(case_id, settings["smoke_candidates"])
        record()
    print(f"Qualified {report['qualified']} of {report['attempted']} attempted seeds. "
          f"Sweep: {report['sweep_status']}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-seeds", type=int, default=5)
    args = parser.parse_args()
    if args.max_new_seeds < 1:
        parser.error("--max-new-seeds must be positive")
    import fcntl
    import torch

    if not torch.cuda.is_available():
        parser.error("the preregistered pilot requires CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pilot.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("a pilot process is already running in this checkout")
        execute(args.max_new_seeds)


if __name__ == "__main__":
    main()
