# Mangafication

Local CPU-first pipeline for turning photographs into strictly two-tone drawing references. Option B is implemented as four deterministic baselines. Option A is planned as a learned interpretation; **no learned model has been trained**.

## Setup

Use the existing `.venv` if available. Required packages: `numpy`, `Pillow`, `opencv-python-headless`, `scikit-image`, `gradio`, `pandas`, `pyarrow`.

```bash
source .venv/bin/activate
```

## Selected image workflow

`training_selection.csv` contains 100 Mugler entries from the Vogue Runway metadata. The scripts read its actual `url` column and preserve all metadata in `data/raw/manifest.jsonl`. Download requests use HTTPS, timeouts, retries, a delay between requests, and image validation. Failed requests remain in the manifest and can be retried by rerunning the command. Existing valid images are reused. Follow the source's access and reuse terms; the downloaded photos are excluded from Git.

```bash
python src/download_images.py
python src/prepare_dataset.py --max-side 1536 --seed 42
```

The second command preserves aspect ratio, exports normalized PNGs and separate shadow and contour masks, generates comparison sheets, and writes train/validation/test JSONL lists. Exact duplicates and perceptual near duplicates are assigned to the same split. Inspect small or unusually similar groups manually before training. Raw images are never resized in place.

## Baseline and UI

```bash
python src/baseline.py data/raw/mugler/EXAMPLE.jpg --ink 0.45 --contour 0.55 --detail 0.55
python app.py
```

The UI opens at `http://127.0.0.1:7860`. It compares global thresholding, adaptive thresholding, Lab color-aware processing, and shadow-plus-contour composition. It exports binary result and mask PNGs. Every output channel is exactly 0 or 255. Black means ink; white means paper.

## Learned Option A plan

Runway photos are unpaired inputs, not manga labels. Collect artist-created or artist-approved binary targets with agreed conventions for shadows, lines, fabric and faces. Begin with a pretrained lightweight vision encoder and a small decoder that predicts separate shadow and contour logits; combine them with an explicit binary threshold. The deterministic masks can be conditioning inputs or weak supervision, but should not be treated as ground truth. Split by duplicate group before annotation or training. Start with frozen encoder features and small crops on the laptop; move full-resolution training to the NVIDIA desktop. Evaluate with held-out photographs using structure preservation and artist review, including comparison against Option B.

## Option B semantic prototype

The earlier thresholds miss black clothing against a black background. The optional `semantic` mode uses a pretrained 18-class human/clothing parser to preserve the person and garment regions, then applies class-aware binary shadow thresholds, high-resolution contours, and a white relief line where black clothing meets a black backdrop. This is a structural prototype; unusual runway pieces and fine feathers still need review and possibly manual correction.

Install only the new CPU runtime into the existing environment, then download the pinned model into ignored local data:

```bash
python -m pip install onnxruntime
python src/download_model.py
python src/baseline.py path/to/photo.jpg --semantic
python src/prepare_dataset.py --max-side 1024 --semantic
```

The last command batch-exports `data/option_b/`, `data/masks/semantic_shadows/`, and `data/masks/semantic_contours/`, along with comparison sheets and the split files. The Gradio UI offers semantic mode when the model is present. The model is [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes), revision `584abc1e1d260e23c0fc627c5217a09b2b461046`; review its model card and terms before distributing the model or outputs. Weights are never committed to Git.
