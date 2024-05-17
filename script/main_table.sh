#! /bin/bash

# citeseer
python buffer.py \
--dataset=citeseer \
--model=sgc2-lr3-wt54 \
--model_name=SGC2 \
--num_experts=100 \
--lr_teacher=1e-3
# 0.25
python condense.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=6000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=5e-7 --reduction_rate=0.25 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=0.999 --tau=0.9 --gamma=0.5 \
--message_passing=4 --dropout_test=0.0
# 0.5
python condense.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=1000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=0.999 --tau=0.9 --gamma=0.5 \
--message_passing=4 --dropout_test=0.0
# 1
python condense.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=100 --eval_interval=10 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=5e-8 --reduction_rate=1 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=10 --beta=0.999 --tau=0.99 --gamma=0.5 \
--message_passing=4 --dropout_test=0.0

# cora
python buffer.py \
--dataset=cora \
--model=sgc2-lr3-wt54 \
--model_name=SGC2 \
--num_experts=100 \
--lr_teacher=1e-3
# 0.25
python condense.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=8000 --eval_interval=200 --student_epochs=5 --max_start_epoch=50 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.25 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=0.9 --tau=0.9 --gamma=0.5 \
--message_passing=3 --dropout_test=0.0
# 0.5
python condense.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=7000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0
# 1
python condense.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=5000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=1 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=0.1 --beta=0.999 --tau=0.9 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0

# ogbn-arxiv
python buffer.py \
--dataset=ogbn-arxiv \
--model=sgc2-lr52-wt54 \
--model_name=SGC2 \
--nlayers=2 \
--num_experts=100 \
--lr_teacher=5e-2
# 1e-3
python condense.py --gpu_id=0 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=3000 --eval_interval=200 --student_epochs=100 --max_start_epoch=60 \
--expert_epochs=1 --lr_feat=1e-2 --reduction_rate=1e-3 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=0.1 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0
# 5e-3
python condense.py --gpu_id=0 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=25000 --eval_interval=500 --student_epochs=100 --max_start_epoch=60 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=5e-3 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=10 --beta=0.9 --tau=0.99 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0
# 1e-2
python condense.py --gpu_id=0 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=100 --max_start_epoch=90 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=1e-2 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=10 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=1 --dropout_test=0.0

# flickr
python buffer.py \
--dataset=flickr \
--model=sgc2-lr3-wt54 \
--model_name=SGC2 \
--num_experts=100 \
--lr_teacher=1e-3
# 1e-3
python condense.py --gpu_id=0 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=50 --max_start_epoch=50 \
--expert_epochs=1 --lr_feat=5e-5 --reduction_rate=1e-3 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=0.999 --tau=0.9 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0
# 5e-3
python condense.py --gpu_id=0 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=10000 --eval_interval=500 --student_epochs=50 --max_start_epoch=95 \
--expert_epochs=2 --lr_feat=5e-5 --reduction_rate=5e-3 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=500 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0
# 1e-2
python condense.py --gpu_id=0 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=10000 --eval_interval=500 --student_epochs=100 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=1e-4 --reduction_rate=1e-2 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1000 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=2 --dropout_test=0.0

# reddit
python buffer.py \
--dataset=reddit \
--model=sgc2-lr3-wt54 \
--model_name=SGC2 \
--num_experts=100 \
--lr_teacher=1e-3
#5e-4
python condense.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=14000 --eval_interval=500 --student_epochs=20 --max_start_epoch=50 \
--expert_epochs=1 --lr_feat=1e-3 --reduction_rate=5e-4 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=500 --beta=0.1 --tau=0.99999 --gamma=0.5 \
--message_passing=3
#1e-3
python condense.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=10000 --eval_interval=500 --student_epochs=20 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=1e-3 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=0.1 --tau=0.99999 --gamma=0.5 \
--message_passing=3
#2e-3
python condense.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=10000 --eval_interval=500 --student_epochs=20 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=1e-2 --reduction_rate=2e-3 --saved_folder=saved_ours \
--normalize --with_val --exps=5 \
--alpha=1 --beta=0.1 --tau=0.99999 --gamma=0.5 \
--message_passing=3
