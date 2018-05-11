# Usage:
# cd slim
# ./gender_export_inf_graph.sh

python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/output/inception_v3_inf_graph.pb \
  --dataset_name=gender 

