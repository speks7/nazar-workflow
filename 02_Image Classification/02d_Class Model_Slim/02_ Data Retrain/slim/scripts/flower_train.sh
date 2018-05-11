#!/bin/bash
# ==============================================================================
# @author: Walton TWang

# flower data training 
# Usage:
# cd slim
# ./slim/scripts/flower_train.sh

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir='D:\TF_Try\tensorflow_models\research\slim\flower\train' \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir='D:\TF_Try\tensorflow_models\research\slim\flower\data' \
  --model_name=inception_v3 \
  --clone_on_cpu=True
  --checkpoint_path='D:\TF_Try\tensorflow_models\research\slim\flower\pretrained-checkpoint/inception_v3.ckpt' \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004