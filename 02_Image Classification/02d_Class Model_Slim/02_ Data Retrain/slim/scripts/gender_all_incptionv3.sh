# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/checkpoints

# Where the dataset is saved to.
DATASET_DIR=/home/twang/Documents/tensorflow-classfication2/gender-airport-entrance

# the(fine-tuned) some last layers
TRAIN_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/training_last_layer

Eval_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/eval_last_layer

# training from scratch
TRAIN_all_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/training_all_layers

Eval_All_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/eval_all_layers


# Fine-tune only the new layers for 100000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/inception_v3 \
  --dataset_name=gender \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --preprocessing_name=gender \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=64 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/inception_v3 \
  --eval_dir=${Eval_DIR}/inception_v3 \
  --dataset_name=gender \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --preprocessing_name=gender 


# Fine-tune all the new layers for 50000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_all_DIR}/inception_v3 \
  --dataset_name=gender \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --preprocessing_name=gender \
  --checkpoint_path=${TRAIN_DIR}/inception_v3 \
  --max_number_of_steps=50000 \
  --batch_size=64 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_all_DIR}/inception_v3 \
  --eval_dir=${Eval_All_DIR}/inception_v3 \
  --dataset_name=gender \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --preprocessing_name=gender


