# AIPR Deepfake Workshop

## Canonical tutorial

Use [Deepfake_Generation_Workshop.ipynb](Deepfake_Generation_Workshop.ipynb). It is the maintained tutorial and has been tested on the Linux RTX 6000 Ada host with Python 3.10 and CUDA 12.4. The v5 notebook remains available as a historical compatibility version.

## Setup

Create the main environment:

```bash
conda create -n aipr python=3.10 -y
conda activate aipr
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements-linux.txt
conda install -n aipr -c conda-forge ffmpeg -y
python -m ipykernel install --user --name aipr --display-name "Python 3.10 (aipr)"
```

For macOS Apple Silicon, replace `onnxruntime-gpu` with `onnxruntime` and install the
macOS-compatible PyTorch wheel:

```bash
python -m pip uninstall -y onnxruntime-gpu
python -m pip install onnxruntime
```

The notebook clones missing official repositories with a bounded timeout. It does not install
their conflicting dependencies into `aipr`; use a separate environment for PhotoMaker:

```bash
conda create -n photomaker python=3.10 -y
conda run -n photomaker python -m pip install torch==2.1.2
conda run -n photomaker python -m pip install -r PhotoMaker/requirements.txt
conda run -n photomaker python -m pip install "diffusers==0.30.0" "transformers==4.44.2" "huggingface-hub==0.24.6" "peft==0.12.0" einops onnxruntime
```

Download LivePortrait weights explicitly before enabling its expensive inference cell:

```bash
python -m pip install "huggingface_hub==0.24.6"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='KlingTeam/LivePortrait', local_dir='LivePortrait/pretrained_weights', ignore_patterns=['*.git*', 'README.md', 'docs/**'])"
WORKSHOP_RUN_EXPENSIVE=1 jupyter nbconvert --to notebook --execute Deepfake_Generation_Workshop_v6_Technical.ipynb --output executed.ipynb
```

In VS Code, select **Kernel -> Select Another Kernel -> Python 3.10 (aipr)**. Set
`WORKSHOP_RUN_EXPENSIVE=1` only after verifying the checkpoint files and consented inputs.

Verify the main environment:

```bash
conda run -n aipr python --version
conda run -n aipr python -c "import numpy, matplotlib, PIL, cv2, torch, transformers, diffusers, accelerate, insightface, onnxruntime; print('aipr imports ok')"
conda run -n aipr jupyter kernelspec list
```

## Tested status

- **LivePortrait:** GPU-tested on the RTX 6000 Ada with CUDA 12.4 and the bundled portrait/driving clip. The three outputs are written under `LivePortrait/out_relative`, `LivePortrait/out_absolute`, and `LivePortrait/out_amplified`.
- **PhotoMaker:** separate environment and large model download required; no identity generation is run without consented reference images.
- **InstantID:** CUDA/SDXL-oriented; the notebook reports GPU-only or incomplete status rather than claiming MPS success.
- **FaceFusion:** repository/CLI audit is bounded; swap inference requires reviewed, consented source and target media.
- **SimSwap:** CUDA-oriented and requires separate pretrained weights; skipped on unsupported devices.
- **DreamID:** research reference unless an official runnable entry point and weights are verified.

The notebook's final gallery displays only non-empty generated files and prints the method,
device, input, and parameter provenance that should be recorded for each artifact. The v6
notebook also writes `workshop_results_v6/artifact_manifest.json` with SHA-256 hashes.

### Verified Linux GPU run

The tested `aipr` environment uses `torch==2.5.1` and `torchvision==0.20.1` from the
CUDA 12.4 PyTorch index. On the workshop host, `torch.cuda.is_available()` returned
`True` for an NVIDIA RTX 6000 Ada with driver 550.163.01. LivePortrait completed all
three guarded variants and produced six non-empty MP4 artifacts (animated and concat
outputs). The remaining methods remain explicitly skipped or audit-only unless their
separate dependencies, checkpoints, and consented inputs are provided.

## PhotoMaker fallback

PhotoMaker's repository documents Mac M1/M2 support:

```bash
conda create -n photomaker python=3.10 -y
conda activate photomaker
python -m pip install --upgrade pip
python -m pip install torch==2.1.2
python -m pip install --no-deps "diffusers==0.30.0" "transformers==4.44.2" "huggingface-hub==0.24.6" "peft==0.12.0" "tokenizers==0.19.1" "numpy==1.26.4" "scipy==1.13.1"
python -m pip install -r PhotoMaker/requirements.txt
python -m pip install einops onnxruntime
python -m pip install git+https://github.com/TencentARC/PhotoMaker.git
```


