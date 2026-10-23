# Ten-photo visual review set

Run `.venv/bin/python src/build_review_set.py` after generating the raw A/B images. The selected source files and current baseline outputs are listed in `selection.json`. Each sheet compares the photograph, deterministic Option B, generated Option A, and generated Option B. Generated images and sheets live under `data/review_targets/` and are excluded from Git because they derive from runway photographs.

The two generated variants used the original photo as the edit target and the user-supplied `mugler-mangafied.jpg` as a style reference. Prompts asked for one full-length rendering with the source pose, anatomy, garments, and composition intact, only black and white pixels, and no text, border, halftone, or gray. Option A emphasized expressive ink masses and linework; Option B emphasized geometric fidelity. The `src/build_review_set.py` pass converts each generated image to exact 0/255 pixel values with a fixed threshold of 160. Raw generations are retained for visual inspection.

These are **synthetic visual references**, not paired training targets or exact ground truth. The image generator sometimes changes garment construction, facial structure, pose, or background details even when asked for fidelity. Review the original photo before accepting any apparent improvement. Do not use pixel similarity to these images as a quality score.

Initial review priorities: fix large, irregular white regions in dark backgrounds; keep black garments separated from dark backgrounds; preserve pale garment folds and sculptural accessory outlines; avoid turning patterned cloth into noise. The `0015571`, `0040264`, and `0006265` sheets expose the background issue most clearly.
