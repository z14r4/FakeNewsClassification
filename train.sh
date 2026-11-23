#!/bin/sh
#BSUB -J fakenews_mm
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -R "span[hosts=1]"
#BSUB -N
#BSUB -o fakenews_mm.%J.out
#BSUB -e fakenews_mm.%J.err

# --- environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
module load cuda/11.8

cd /work3/s225083/indeng242/fake_news

# Make sure transformers is installed (other deps already in base env).
python -c "import transformers" 2>/dev/null \
    || pip install --user --quiet "transformers>=4.40" "huggingface_hub>=0.23"

# Cache HF model files locally so the compute node doesn't need internet
# the second time. First run will download bert-base-uncased.
export HF_HOME="${HF_HOME:-/work3/s225083/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

# --- wandb ---
# Run `wandb login` on the LOGIN node once before submitting so the API key
# is in ~/.netrc. If you can't do that, set WANDB_API_KEY here, or change
# --wandb_mode to "offline".
export WANDB_DIR=/work3/s225083/indeng242/fake_news/runs
mkdir -p "$WANDB_DIR"

# --- info ---
echo "==== job $LSB_JOBID on $(hostname) ===="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# --- train ---
python train.py \
    --train_csv manifests/train.csv \
    --test_csv  manifests/test.csv \
    --image_root . \
    --bert bert-base-uncased \
    --epochs 100 \
    --patience 0 \
    --batch_size 32 \
    --max_len 64 \
    --image_size 224 \
    --lr_bert 2e-5 \
    --lr_other 1e-3 \
    --num_workers 4 \
    --wandb_project fake-news-multimodal \
    --wandb_run_name "bert+dino-${LSB_JOBID}" \
    --out_dir runs

echo "==== done ===="
