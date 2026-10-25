# Demo app: how to run it on your own computer

This folder holds the project's web app, built with Gradio. It runs both stages in one browser page:

- **Stage 1**: pick or draw a 128 × 128 plan, mark a region, and the model redraws it.
- **Stage 2**: pick or upload a real floor plan drawing. The system finds what is missing and draws it in
  (PARSE → CHECK → FILL → VERIFY), and every step is shown. There is also a manual mode where you mark the region
  yourself.
- **How it works · results**: the method and the test numbers (`ABOUT.md`).

| file | what it is |
|---|---|
| `app.py` | the page layout and the button handlers. All the model code is in `src/stage1/interface/` and `src/stage2/interface/` |
| `ABOUT.md` | the text of the "How it works · results" tab |

## Why it is not hosted online (Hugging Face Spaces)

We tried to host the app on a Hugging Face Space so it could be opened from a link, but it was not possible:

- **Model size**: the trained models are about 1.6 GB (10 checkpoints), and the app loads about 1.5 GB of them into
  memory on the first Stage 2 run.
- **No GPU on the free tier**: a free Space runs on a small CPU only. Stage 2 takes 1–2 minutes per drawing on a
  laptop CPU, and longer on a shared Space, so it times out or is unusable.
- **Data licence**: the example drawings come from CubiCasa5K (CC BY-NC 4.0) and ResPlan. We do not redistribute
  them on a public page or in the public GitHub repository.

So the app is run **locally**. It needs no GPU: everything below works on an ordinary laptop with 8 GB of RAM or
more, on Windows, macOS or Linux.

## What you need

| what | where to get it | size |
|---|---|---|
| the code | GitHub: https://github.com/venkyc007/floorplan-diffusion | small |
| the trained models | Kaggle dataset `venkatesh007c/floorplan-stage2-final`: https://www.kaggle.com/datasets/venkatesh007c/floorplan-stage2-final | 1.6 GB |
| the example drawings / plans (optional, recommended) | `demo_app_examples.zip` in the submission Google Drive folder | 5 MB |
| Python 3.11 | Miniconda: https://docs.conda.io/en/latest/miniconda.html (or any Python 3.11) | |

Without the examples zip the app still works, but the two galleries are empty. You can then draw a plan in the
Stage 1 editor or upload any floor plan image in Stage 2.

## Step by step

### 1. Get the code

Either run `git clone https://github.com/venkyc007/floorplan-diffusion.git`, or open the GitHub page, click
**Code → Download ZIP** and unzip it. In the steps below, the **project folder** is the folder that contains
`README.md`, `requirements.txt`, `src/` and `demo_app/`.

### 2. Put the trained models in place

1. Open the Kaggle dataset link above (a free Kaggle account is needed to download) and click **Download**. You get a
   zip, usually `archive.zip`.
2. Unzip it. Inside is a folder `checkpoints/`. If Kaggle delivered it as `checkpoints.zip`, unzip that too. The
   `wheels/` folder is not needed.
3. Inside `checkpoints/` is a folder called `outputs/`. **Copy this `outputs` folder into the project folder.** It
   merges with the `outputs/` folder that is already there. Answer "yes / merge" if asked; no existing file is
   replaced.

Check: these two files must now exist in the project folder:

```
outputs/kaggle/parser_v2/runs/parser_v2B/checkpoints/best.pt
outputs/runs/unet_matched/checkpoints/best.pt
```

### 3. Put the examples in place (optional)

Copy `demo_app_examples.zip` from the Google Drive folder into the **project folder** and unzip it **there**. It
creates `data/app/`, which holds the Stage 1 gallery (`val_gallery.npz`), the Stage 2 gallery
(`stage2_gallery/`) and a few files to try the upload with (`test_inputs/`).

### 4. Install Python and the packages (once)

Open a terminal in the project folder. On Windows use "Anaconda Prompt (Miniconda3)" from the Start menu, then
`cd` into the project folder. Then run:

```bash
conda create -n floorplan python=3.11 -y
conda activate floorplan
pip install -r requirements.txt
```

This downloads about 1–2 GB of packages (PyTorch, Gradio, …) and takes a few minutes. Without conda, a plain
`python -m venv .venv`, then activating it and running `pip install -r requirements.txt`, also works. It must be
**Python 3.11**.

Optional check: `python -m pytest tests/` should report **246 passed** (about 1 minute).

### 5. Start the app

From the project folder, with the environment active:

```bash
python demo_app/app.py
```

Wait until the terminal prints a line like `Running on local URL:  http://127.0.0.1:7860`. Then open
**http://127.0.0.1:7860** in a browser. To stop the app, press `Ctrl + C` in the terminal.

Each time after that, only `conda activate floorplan` and `python demo_app/app.py` are needed.

### 6. Try it

- **Stage 1 tab**: click a plan in the gallery. Then press **Benchmark mask** or **Random mask**, or brush a
  region. Then press **Repair**. You see the input, the repaired plan, the difference, and scores against the true
  plan.
- **Stage 2 tab**: click a drawing in the gallery. The caption says what was erased. Then:
  - Press **2a · Automatic** and wait about 1–2 minutes on a CPU (the first run also loads the models). You see the
    parsed map, what CHECK flagged, the filled map, and the additions in red on the drawing.
  - Open **Show the answer** to see the erased spot.
  - For a missed part, use **2b manual**: brush the missing region, then press **Fill the marked region(s)**.
  - You can also upload any black-and-white floor plan image.

## If something goes wrong

| message / problem | fix |
|---|---|
| "checkpoint missing" / `CheckpointMissing` | the `outputs` folder from step 2 is not in the right place; check the two paths in step 2 |
| "No sample gallery found" / "No examples yet" | step 3 was skipped. The app still works with the editor and uploads |
| `ModuleNotFoundError` | the environment is not active (`conda activate floorplan`), or step 4 did not finish |
| port 7860 already in use | another copy of the app is running; close it, or change `server_port` in `configs/stage1/app.yaml` |
| Stage 2 is slow | expected on a CPU: about 90 s automatic, 60–120 s manual per drawing. With an NVIDIA GPU and CUDA PyTorch, set `STAGE2_DEVICE=cuda` before starting |

The app's settings are frozen: they were chosen on validation data, never on what the app shows or on test data.
They are in `configs/stage1/app.yaml`, `configs/stage2/app.yaml` and `configs/stage2/pipeline.yaml`. More detail on
the interface: `docs/interface.md`.
