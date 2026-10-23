"""Download the optional CPU clothing parser; model weights stay out of Git."""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = 'mattmdjaga/segformer_b2_clothes'
REVISION = '584abc1e1d260e23c0fc627c5217a09b2b461046'
DESTINATION = Path(__file__).resolve().parents[1] / 'data/models'

if __name__ == '__main__':
    for name in ('onnx/model.onnx', 'onnx/config.json', 'onnx/preprocessor_config.json'):
        print(hf_hub_download(REPO, name, revision=REVISION, local_dir=DESTINATION))
