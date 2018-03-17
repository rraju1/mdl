#! /bin/bash

cd /research/rraju2/mdl/tensorflow_scripts
source activate tensorflow
#for lambda in 100000 10000 1000
#do
#echo $lambda
python feedforward.py
for trial in {1..20}
do
for number in {0..20..1}
do
echo $number
python err_infer.py --err=$number
done
done
#done
source deactivate
exit 0
