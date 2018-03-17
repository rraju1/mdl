#! /bin/bash

cd /research/rraju2/mdl/tensorflow_scripts/
source activate tensorflow
for trial in {1..20}
do
for number in {0..20..1}
do
echo $number
python err_ori_infer.py --err=$number
done
done
#done
source deactivate
exit 0
