#!/usr/bin/env bash

# This script trains FastPitch on the Vivos dataset.
export OMP_NUM_THREADS=1

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=16}
: ${GRAD_ACCUMULATION:=2}
: ${OUTPUT_DIR:="./output_vivos"}
: ${LOG_FILE:=$OUTPUT_DIR/nvlog.json}
: ${DATASET_PATH:="data/vivos"}
: ${TRAIN_FILELIST:=filelists/vivos_train.txt}
: ${VAL_FILELIST:=filelists/vivos_test.txt}
: ${AMP:=true}

: ${LEARNING_RATE:=0.1}
: ${EPOCHS:=1000}
: ${EPOCHS_PER_CHECKPOINT:=50}

# Vivos specific settings
: ${TEXT_CLEANERS:=basic_cleaners}
: ${SYMBOL_SET:=vietnamese_basic}

# Fastpitch configurations
: ${PHONE:=false}
: ${ENERGY:=true}
: ${LOAD_PITCH_FROM_DISK:=true}
: ${LOAD_MEL_FROM_DISK:=false}

# Vivos has 1 speaker per utterance but many speakers in total. 
# Since we didn't add speaker embedding in the filelist processing, we treat it as 1 speaker or we can add it. 
# For simplicity, treating as 1 speaker.
: ${NSPEAKERS:=1}

ARGS=""
ARGS+=" --cuda"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --dataset-path \"$DATASET_PATH\""
ARGS+=" --training-files $TRAIN_FILELIST"
ARGS+=" --validation-files $VAL_FILELIST"
ARGS+=" -bs $BATCH_SIZE"
ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"
ARGS+=" --optimizer lamb"
ARGS+=" --epochs $EPOCHS"
ARGS+=" --epochs-per-checkpoint $EPOCHS_PER_CHECKPOINT"
ARGS+=" -lr $LEARNING_RATE"
ARGS+=" --weight-decay 1e-6"
ARGS+=" --grad-clip-thresh 1000.0"
ARGS+=" --dur-predictor-loss-scale 0.1"
ARGS+=" --pitch-predictor-loss-scale 0.1"
ARGS+=" --trainloader-repeats 1"
ARGS+=" --validation-freq 10"
ARGS+=" --kl-loss-start-epoch 0"
ARGS+=" --kl-loss-warmup-epochs 100"

ARGS+=" --text-cleaners $TEXT_CLEANERS"
ARGS+=" --symbol-set $SYMBOL_SET"
ARGS+=" --n-speakers $NSPEAKERS"

[ "$AMP" = "true" ]                    && ARGS+=" --amp"
[ "$PHONE" = "true" ]                  && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]                 && ARGS+=" --energy-conditioning"
[ "$LOAD_MEL_FROM_DISK" = true ]       && ARGS+=" --load-mel-from-disk"
[ "$LOAD_PITCH_FROM_DISK" = true ]     && ARGS+=" --load-pitch-from-disk"
[[ "$ARGS" != *"--checkpoint-path"* ]] && ARGS+=" --resume"

mkdir -p "$OUTPUT_DIR"

DISTRIBUTED="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"

python $DISTRIBUTED train.py $ARGS "$@"
