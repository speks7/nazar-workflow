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


# Fine-tune all layers for 100000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_all_DIR}/vgg_16 \
  --dataset_name=gender \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --preprocessing_name=gender_vgg16 \
  --checkpoint_path=${TRAIN_all_DIR}/vgg_16 \
  --max_number_of_steps=50000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --log_every_n_steps=10 \
  --optimizer=sgd \
  --weight_decay=0.0005 


