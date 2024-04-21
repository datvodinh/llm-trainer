pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install .
pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
# Temporary fix for https://github.com/huggingface/datasets/issues/6753
pip install datasets==2.16.0 fsspec==2023.10.0 gcsfs==2023.10.0