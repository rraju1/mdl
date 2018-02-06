#! /bin/bash

source activate tensorflow
python feedforward.py
for number in {0..100..5}
do
echo $number
python freeze.py --model_dir=/research/rraju2/mdl/tensorflow_scripts/results/ --output_node_names=Mean_1 --err=$number
# cd /research/rraju2/tensorflow
# bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
#   --in_graph=/research/rraju2/mdl/tensorflow_scripts/results/frozen_model.pb \
#   --out_graph=/research/rraju2/mdl/tensorflow_scripts/results/costQuanNet.pb \
#   --inputs=Placeholder \
#   --outputs=Mean_1 \
#   --transforms='add_default_attributes
#     fold_constants(ignore_errors=true)
#     fold_batch_norms fold_old_batch_norms quantize_weights
#     quantize_nodes sort_by_execution_order'
# cd -
python load_graph.py
done
source deactivate
exit 0
