#!/usr/bin/env bash

set -e

# Đường dẫn tới dataset
: ${DATA_DIR:="data/vivos"}

# File danh sách
: ${TRAIN_FILELIST:="filelists/vivos_train.txt"}
: ${TEST_FILELIST:="filelists/vivos_test.txt"}

# Cấu hình tiếng Việt
: ${SYMBOL_SET:="vietnamese_basic"}
: ${TEXT_CLEANERS:="basic_cleaners"}

python prepare_dataset.py \
    --wav-text-filelists $TRAIN_FILELIST $TEST_FILELIST \
    --dataset-path "$DATA_DIR" \
    --extract-mels \
    --extract-pitch \
    --save-alignment-priors \
    --f0-method pyin \
    --n-workers 8 \
    --batch-size 1 \
    --sampling-rate 16000 \
    --symbol_set $SYMBOL_SET \
    --text-cleaners $TEXT_CLEANERS
