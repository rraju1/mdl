#! /bin/bash

cd /research/rraju2/mdl/tensorflow_scripts/
source activate tensorflow
#for lambda in 100000 10000 1000
#do
#echo $lambda
#python feedforward.py --lambda_term=10000
for trial in {1..40}
do
for number in {0..20..2}
do
echo $number
python err_ori_infer.py --err=$number
done
done
#done
source deactivate
exit 0
