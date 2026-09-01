#!/usr/bin/env bash

# This script runs inference using the trained FastPitch model on Vivos.
export OMP_NUM_THREADS=1

# Change this to your desired checkpoint
CHECKPOINT_DIR="./output_vivos"
FASTPITCH_CKPT=$(ls -t ${CHECKPOINT_DIR}/FastPitch_checkpoint_*.pt | head -n 1)

if [ -z "$FASTPITCH_CKPT" ]; then
    echo "No checkpoint found in ${CHECKPOINT_DIR}."
    exit 1
fi

echo "Using checkpoint: $FASTPITCH_CKPT"

# The text to synthesize. Must be in Vietnamese.
INPUT_TEXT="phrases/vivos_phrases.tsv"

# Create a sample phrases file if it doesn't exist
mkdir -p phrases
if [ ! -f "$INPUT_TEXT" ]; then
    echo "đây là một ví dụ về tổng hợp tiếng việt" > "$INPUT_TEXT"
    echo "hôm nay trời rất đẹp" >> "$INPUT_TEXT"
fi

OUTPUT_DIR="./output_vivos_audio"
mkdir -p "$OUTPUT_DIR"

# WaveGlow checkpoint for vocoder (you may need to download or use HiFi-GAN)
# In this example, we assume we just generate mel-spectrograms if vocoder is not provided.
# To generate audio, you'll need to specify --hifigan <path_to_hifigan.pt> or --waveglow <path_to_waveglow.pt>

python inference.py \
    --cuda \
    --fastpitch "$FASTPITCH_CKPT" \
    -i "$INPUT_TEXT" \
    -o "$OUTPUT_DIR" \
    --text-cleaners basic_cleaners \
    --symbol-set vietnamese_basic \
    --batch-size 1 \
    --denoising-strength 0.01 \
    --amp
