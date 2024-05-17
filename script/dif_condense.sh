#! /bin/bash
# cora

# gcn
python buffer.py \
--dataset=cora \
--model=gcn-lr3-wt54 \
--model_name=GCN \
--num_experts=100 \
--lr_teacher=1e-3
# 0.5
python condense.py --gpu_id=0 --dataset=cora --expert_net=gcn-lr3-wt54 --expert_net_type=GCN \
--test_net_type=GCN --epochs=7000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_difarch/gcn \
--normalize --with_val --exps=5 \
--alpha=1 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=3 --dropout_test=0.0


# appnp
python buffer.py \
--dataset=cora \
--model=appnp-lr3-wt54 \
--model_name=APPNP \
--num_experts=100 \
--lr_teacher=1e-3
# 0.5
python condense.py --gpu_id=0 --dataset=cora --expert_net=appnp-lr3-wt54 --expert_net_type=APPNP \
--test_net_type=GCN --epochs=4000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_difarch/graphsage \
--normalize --with_val --exps=5 \
--alpha=1 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=4 --dropout_test=0.0


# graphsage
python buffer.py \
--dataset=cora \
--model=graphsage-lr3-wt54 \
--model_name=GraphSage \
--num_experts=100 \
--lr_teacher=1e-3
# 0.5
python condense.py --gpu_id=0 --dataset=cora --expert_net=graphsage-lr3-wt54 --expert_net_type=GraphSage \
--test_net_type=GCN --epochs=8000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_difarch/graphsage \
--normalize --with_val --exps=5 \
--alpha=1 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=3 --dropout_test=0.0


# cheby
python buffer.py \
--dataset=cora \
--model=cheby-lr3-wt54 \
--model_name=Cheby \
--num_experts=100 \
--lr_teacher=1e-3
# 0.5
python condense.py --gpu_id=0 --dataset=cora --expert_net=cheby-lr3-wt54 --expert_net_type=Cheby \
--test_net_type=GCN --epochs=7000 --eval_interval=200 --student_epochs=5 --max_start_epoch=60 \
--expert_epochs=2 --lr_feat=1e-6 --reduction_rate=0.5 --saved_folder=saved_difarch/cheby \
--normalize --with_val --exps=5 \
--alpha=1 --beta=1 --tau=0.9 --gamma=0.5 \
--message_passing=4 --dropout_test=0.0
