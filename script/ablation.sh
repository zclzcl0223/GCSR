#! /bin/bash

# citeseer
# MTT
# 0.25
python mtt.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=3000 --eval_interval=100 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.25 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 0.5
python mtt.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=6000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=5e-7 --reduction_rate=0.5 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 1.0
python mtt.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=3000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=5e-7 --reduction_rate=1 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# Message passing initialization
# 0.25
python mtt.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=3000 --eval_interval=100 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.25 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=4 --dropout_test=0.0
# 0.5
python mtt.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=6000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=5e-7 --reduction_rate=0.5 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=4 --dropout_test=0.0
# 1
python mtt.py --gpu_id=0 --dataset=citeseer --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=1000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=5e-7 --reduction_rate=1 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=4 --dropout_test=0.0


# cora
# MTT
# 0.25
python mtt.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=8000 --eval_interval=200 --student_epochs=5 --max_start_epoch=50 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.25 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 0.5
python mtt.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=7000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 1
python mtt.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=7000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=1 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# Message passing initialization
# 0.25
python mtt.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=8000 --eval_interval=200 --student_epochs=5 --max_start_epoch=50 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.25 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=3 --dropout_test=0.0
# 0.5
python mtt.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=7000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0
# 1
python mtt.py --gpu_id=0 --dataset=cora --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=5000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=1 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0


# ogbn-arxiv
# MTT
# 1e-3
python mtt.py --gpu_id=1 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=15000 --eval_interval=500 --student_epochs=100 --max_start_epoch=60 \
--expert_epochs=1 --lr_feat=1e-2 --reduction_rate=1e-3 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 5e-3
python mtt.py --gpu_id=1 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=100 --max_start_epoch=60 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=5e-3 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 1e-2
python mtt.py --gpu_id=1 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=100 --max_start_epoch=90 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=1e-2 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# Message passing initialization
# 1e-3
python mtt.py --gpu_id=0 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=2000 --eval_interval=200 --student_epochs=100 --max_start_epoch=60 \
--expert_epochs=1 --lr_feat=1e-2 --reduction_rate=1e-3 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0
# 5e-3
python mtt.py --gpu_id=1 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=100 --max_start_epoch=60 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=5e-3 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0
# 1e-2
python mtt.py --gpu_id=1 --dataset=ogbn-arxiv --expert_net=sgc2-lr52-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=100 --max_start_epoch=90 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=1e-2 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=1 --dropout_test=0.0


# flickr
# MTT
# 1e-3
python mtt.py --gpu_id=1 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=15000 --eval_interval=200 --student_epochs=50 --max_start_epoch=50 \
--expert_epochs=1 --lr_feat=5e-5 --reduction_rate=1e-3 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 5e-3
python mtt.py --gpu_id=1 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=15000 --eval_interval=200 --student_epochs=50 --max_start_epoch=95 \
--expert_epochs=2 --lr_feat=1e-4 --reduction_rate=5e-3 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0 --dropout_test=0.0
# 1e-2
python mtt.py --gpu_id=1 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=3000 --eval_interval=200 --student_epochs=100 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=1e-4 --reduction_rate=1e-2 --saved_folder=saved_mtt \
--normalize --with_val --with_figure \
--message_passing=0 --dropout_test=0.0
# Message passing initialization
# 1e-3
python mtt.py --gpu_id=1 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=15000 --eval_interval=200 --student_epochs=50 --max_start_epoch=50 \
--expert_epochs=1 --lr_feat=5e-5 --reduction_rate=1e-3 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0
# 5e-3
python mtt.py --gpu_id=1 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=15000 --eval_interval=200 --student_epochs=50 --max_start_epoch=95 \
--expert_epochs=2 --lr_feat=1e-4 --reduction_rate=5e-3 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0
# 1e-2
python mtt.py --gpu_id=1 --dataset=flickr --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=2000 --eval_interval=200 --student_epochs=100 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=1e-4 --reduction_rate=1e-2 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=2 --dropout_test=0.0


# reddit
# MTT
#5e-4
python mtt.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=15000 --eval_interval=500 --student_epochs=20 --max_start_epoch=50 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=5e-4 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0
#1e-3
python mtt.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=20 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=1e-3 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0
#2e-3
python mtt.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=500 --student_epochs=20 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=5e-3 --reduction_rate=2e-3 --saved_folder=saved_mtt \
--normalize --with_val --exps=5 \
--message_passing=0
# Message passing initialization
#5e-4
python mtt.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=1000 --student_epochs=20 --max_start_epoch=50 \
--expert_epochs=1 --lr_feat=1e-4 --reduction_rate=5e-4 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=3
#1e-3
python mtt.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=1000 --student_epochs=20 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=1e-3 --reduction_rate=1e-3 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=3
#2e-3
python mtt.py --gpu_id=0 --dataset=reddit --expert_net=sgc2-lr3-wt54 --expert_net_type=SGC2 \
--test_net_type=GCN --epochs=20000 --eval_interval=1000 --student_epochs=20 --max_start_epoch=95 \
--expert_epochs=1 --lr_feat=1e-3 --reduction_rate=2e-3 --saved_folder=saved_x \
--normalize --with_val --exps=5 \
--message_passing=3
