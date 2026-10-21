# Demo video — a script (about 6–8 minutes)

The demo shows the experiments working. Start the app beforehand so the models are loaded (first Stage 2 run
loads ~1.5 GB): `python app/app.py`, open http://127.0.0.1:7860, run one Stage 2 example once before recording.
Times on a laptop CPU: Stage 1 U-Net ~1.5 s, hybrid 30–50 s; Stage 2 about 1–2 minutes per drawing (cut the
waiting in editing, or say what is happening while it runs).

| time | what to show | say |
|---|---|---|
| 0:00–0:30 | the page: title, the two tabs, the "How it works" tab | the task; Stage 1 = region given, Stage 2 = find and fill |
| 0:30–2:30 | **Stage 1**: pick a val sample (a door hole) → Repair with the size-adaptive hybrid → input / repaired / difference + scores; switch the method to the matched U-Net and repair again; press **Random mask** twice (a wall, then a room) and repair; brush a mask by hand | the hybrid keeps walls connected (difference image, rule table); scores against the true plan; random masks follow the benchmark rules |
| 2:30–5:00 | **Stage 2 automatic**: pick a gallery drawing with an erased door → 2a → walk through the four step pictures: parsed map, CHECK flags (type + rule), completed map, red additions on the drawing; read the message ("… had no door → door added …") and the score line | PARSE reads 16 classes; CHECK combines rules and detectors; FILL keeps only the flagged type; VERIFY picks the best of N |
| 5:00–6:00 | **a failure, honestly**: a gallery drawing with a whole room erased → 2a finds nothing large → **2b manual**: brush the region → fill | large omissions are not detected automatically; manual mode runs only PARSE + FILL |
| 6:00–6:40 | an **intact** gallery drawing → 2a → any false alarms | false alarms on complete drawings are part of the evaluation |
| 6:40–7:30 | **upload** a floor plan image from outside the dataset (e.g. a scanned plan with a door whited out) → 2a | works on any resolution; results are not scored (no ground truth) |
| 7:30–8:00 | `outputs/stage1/stage1_test.md`, `docs/results.md` tables | where the numbers in the report come from |

Tip: keep the browser zoom at 100% and the window wide, so the four step pictures are readable on the recording.
