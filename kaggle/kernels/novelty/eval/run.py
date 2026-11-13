"""Kaggle kernel floorplan-novelty-eval (GPU T4 x2): evaluations of the novelty work. STAGE picks the job:

  tune          TUNING drawings, map level (src/novelty/eval_maps.py): the locator sweep, then the fills with the
                hole given and the locator -> fill chain, two shards (one per GPU)
  e2e_tuning    the end-to-end comparison on TUNING drawings (src/novelty/eval_e2e.py; a check before the real one)
  e2e_heldout   the end-to-end comparison on the HELD-OUT val drawings (the reported numbers; run once per
                frozen configuration)
CONFIGS: {tag: novelty pipeline config}; every config is evaluated (e.g. with and without the Fourier blocks).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of the Stage 2 kernels (data, parser v2, k2a, detv2,
detv3, k1, b4: the shipped pipeline) and of the novelty kernels (data, locate, fill-ffc, fill-plain).
Output: /kaggle/working/novelty/ (csv + md).
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
STAGE = "e2e_heldout"
CONFIGS = {"_v2": "configs/novelty/pipeline_large_edit.yaml"}
N_TUNE = 120
E2E_LIMIT = None           # drawings per shard (None = all)
EXTRA = '--arms "hybrid auto,hybrid manual" --figure 6'   # extra arguments for the stage's script
SWEEP = False               # tune: also run the locator and gate sweeps


def copy_code():
    hits = [h for h in glob.glob("/kaggle/input/**/src/utils.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


def place_novelty_checkpoints(find):
    """Copy every novelty checkpoint named in the configs to the path the config expects (missing ones are skipped
    with a note: a config whose fill models are missing is not evaluated)."""
    import yaml
    usable = {}
    for tag, path in CONFIGS.items():
        if path is None:                                   # shipped arms only: no novelty models needed
            usable[tag] = None
            continue
        c = yaml.safe_load(open(f"{CODE}/{path}", encoding="utf-8"))["checkpoints"]
        ok = True
        for p in c["locator"] + c.get("structure", []) + [c["unet"], c["diffusion"]]:
            src = find("/".join(p.split("/")[-4:]))                     # runs/<name>/checkpoints/best.pt
            if not src:
                print("MISSING", p, flush=True)
                ok = ok and ((p in c["locator"] and len(c["locator"]) > 1) or
                             (p in c.get("structure", []) and len(c["structure"]) > 1))
                continue
            os.makedirs(os.path.dirname(f"{CODE}/{p}"), exist_ok=True)
            shutil.copy(src[0], f"{CODE}/{p}")
            print(p, "<-", src[0], flush=True)
        if ok:
            usable[tag] = path
    return usable


def both_gpus(cmds, env):
    """cmds: one shell command per GPU, run in parallel; raises if any fails."""
    procs = [subprocess.Popen(c, shell=True, cwd=CODE, env={**os.environ, **env, "CUDA_VISIBLE_DEVICES": str(i),
                                                            "STAGE2_DEVICE": "cuda", "PYTHONUNBUFFERED": "1"})
             for i, c in enumerate(cmds)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"exit codes {codes}")


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import find, run
    from src.stage2.kaggle_cpu import setup
    data = setup()
    configs = place_novelty_checkpoints(find)
    out = f"{CODE}/outputs/novelty"
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    try:
        for tag, cfg in configs.items():
            env = {"NOVELTY_CONFIG": f"{CODE}/{cfg}"} if cfg else {}
            if STAGE == "tune":
                sets = " ".join(sorted(find("large9_tune_*.zip")) or sorted(find("large_tune_*.zip")))
                if SWEEP:
                    both_gpus([f"{py} -m src.novelty.eval_maps --set {sets} --locator-sweep --tag {tag} > {out}/sweep{tag}.log 2>&1",
                               f"{py} -m src.novelty.eval_maps --set {sets} --gate-sweep --tag {tag} > {out}/gate{tag}.log 2>&1"], env)
                both_gpus([f"{py} -m src.novelty.eval_maps --set {sets} --n {N_TUNE} --shard {i}/2 --tag {tag} {EXTRA} "
                           f"> {out}/tune{tag}_{i}.log 2>&1" for i in range(2)], env)
                run(f"{py} -m src.novelty.eval_maps --report '{out}/tune_maps{tag}_*.csv' --tag {tag}")
            else:
                split = STAGE.split("_")[1]
                lim = f"--limit {E2E_LIMIT}" if E2E_LIMIT else ""
                both_gpus([f"{py} -m src.novelty.eval_e2e --zip {data}/parser_val.zip --split {split} --shard {i}/2 "
                           f"--tag {tag} {lim} {EXTRA} > {out}/e2e_{split}{tag}_{i}.log 2>&1" for i in range(2)], env)
                note = ("held-out: never used to choose a setting" if split == "heldout" else
                        "TUNING drawings: used to choose settings, so optimistic")
                run(f"{py} -m src.novelty.eval_e2e --report '{out}/e2e_{split}{tag}_*.csv' --tag _{split}{tag} "
                    f"--split-note '{note}'")
    finally:
        os.makedirs("/kaggle/working/novelty", exist_ok=True)
        for f in glob.glob(f"{out}/*"):
            shutil.copy(f, "/kaggle/working/novelty")
        for f in sorted(glob.glob(f"{out}/*.log")):
            print(f"===== tail of {os.path.basename(f)}", flush=True)
            print("".join(open(f, errors="replace").readlines()[-8:]), flush=True)
