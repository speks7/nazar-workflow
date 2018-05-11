# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/checkpoints

# Where the dataset is saved to.
DATASET_DIR=/home/twang/Documents/tensorflow-classfication2/gender-airport-entrance

# the(fine-tuned) last layers
TRAIN_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/training_last_layer

Eval_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/eval_last_layer


# training from scratch
TRAIN_all_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/training_all_layers

Eval_All_DIR=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/airport_gender3/eval_all_layers

# Fine-tune only the last layers for 100000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/vgg_16 \
  --dataset_name=gender \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --preprocessing_name=gender_vgg16 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt \
  --checkpoint_exclude_scopes=vgg_16/fc8 \
  --trainable_scopes=vgg_16/fc8 \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=sgd \
  --weight_decay=0.0005 

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/vgg_16 \
  --eval_dir=${Eval_DIR}/vgg_16 \
  --dataset_name=gender \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --preprocessing_name=gender_vgg16

# Fine-tune all layers for 100000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_all_DIR}/vgg_16 \
  --dataset_name=gender \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --preprocessing_name=gender_vgg16 \
  --checkpoint_path=${TRAIN_DIR}/vgg_16 \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=sgd \
  --weight_decay=0.0005 

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_all_DIR}/vgg_16 \
  --eval_dir=${Eval_All_DIR}/vgg_16 \
  --dataset_name=gender \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --preprocessing_name=gender_vgg16


