cd /home/twang/Documents/tensorflow/bazel-bin/tensorflow/python/tools/

./freeze_graph \
  --input_graph=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/output/inception_v3_inf_graph.pb \
  --input_checkpoint=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/inception-v3/model.ckpt-1000 \
  --input_binary=true --output_graph=/home/twang/Documents/tensorflow-classfication2/train-slim/trained-models/output/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
