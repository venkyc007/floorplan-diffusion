"""Re-point the evaluation kernel (kaggle/kernels/novelty/eval): its STAGE, CONFIGS, EXTRA and SWEEP constants and the
kernel outputs it attaches (derived from the checkpoint paths in the configs).

    python kaggle/kernels/novelty/point_eval.py e2e_tuning "--arms \"hybrid auto,hybrid manual\"" 0 _v3=configs/novelty/pipeline_structure.yaml
"""
import json
import re
import sys

stage, extra, sweep = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
configs = dict(a.split("=") for a in sys.argv[4:])
p = "kaggle/kernels/novelty/eval/run.py"
s = open(p, encoding="utf-8").read()
s = re.sub(r'^STAGE = .*$', f'STAGE = "{stage}"', s, flags=re.M)
s = re.sub(r'^CONFIGS = .*$', "CONFIGS = " + json.dumps(configs), s, flags=re.M)
s = re.sub(r'^EXTRA = .*$', f"EXTRA = {extra!r}   # extra arguments for the stage's script", s, flags=re.M)
s = re.sub(r'^SWEEP = .*$', f"SWEEP = {sweep}               # tune: also run the locator and gate sweeps", s, flags=re.M)
open(p, "w", encoding="utf-8", newline=chr(10)).write(s)
compile(s, p, "exec")
m = json.load(open("kaggle/kernels/novelty/eval/kernel-metadata.json", encoding="utf-8"))
base = [k for k in m["kernel_sources"] if "novelty" not in k]
nov = ["venkatesh007c/floorplan-novelty-data2", "venkatesh007c/floorplan-novelty-locate"]
import yaml
for cfg in configs.values():
    c = yaml.safe_load(open(cfg, encoding="utf-8"))["checkpoints"]
    for q in c.get("structure", []) + [c["unet"], c["diffusion"]]:
        k = "venkatesh007c/floorplan-" + q.split("/")[2].replace("_", "-")          # outputs/kaggle/novelty_fill_ffc/...
        if k not in nov:
            nov.append(k)
m["kernel_sources"] = base + nov
open("kaggle/kernels/novelty/eval/kernel-metadata.json", "w", encoding="utf-8", newline=chr(10)).write(json.dumps(m, indent=2) + chr(10))
print(stage, configs, m["kernel_sources"][-4:])
