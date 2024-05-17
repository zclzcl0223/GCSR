#! /bin/bash

for dataset in citeseer cora
do
for reduction_rate in 0.25 0.5 1
do
python test.py --gpu_id=1 --dataset=${dataset} --load_folder=saved_ours \
--exps=5 --normalize --with_val --reduction_rate=${reduction_rate}
done
done

for dataset in ogbn-arxiv flickr
do
for reduction_rate in 1e-3 5e-3 1e-2
do
python test.py --gpu_id=1 --dataset=${dataset} --load_folder=saved_ours \
--exps=5 --normalize --with_val --reduction_rate=${reduction_rate}
done
done

for dataset in reddit
do
for reduction_rate in 5e-4 1e-3 2e-3
do
python test.py --gpu_id=1 --dataset=${dataset} --load_folder=saved_ours \
--exps=5 --normalize --with_val --reduction_rate=${reduction_rate}
done
done