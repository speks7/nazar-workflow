#Linux retraining command
python label_image.py \
  --graph retrained_graph.pb \
  --labels retrained_labels.txt \
  --input_layer=Placeholder \
  --output_layer=final_result \
  --image index.jpg