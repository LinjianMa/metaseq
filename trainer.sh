#!/bin/bash

#SBATCH --job-name=metaseq-trainer
#SBATCH --ntasks=16
#SBATCH --nodes=16
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --exclude=a100-st-p4d24xlarge-6,a100-st-p4d24xlarge-7,a100-st-p4d24xlarge-28,a100-st-p4d24xlarge-48,a100-st-p4d24xlarge-16,a100-st-p4d24xlarge-58,a100-st-p4d24xlarge-56,a100-st-p4d24xlarge-5,a100-st-p4d24xlarge-8,a100-st-p4d24xlarge-57


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
# Enable for A100
export LOGLEVEL=INFO
export FI_PROVIDER="efa"
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_ALGO=ring

# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
# srun torchrun --nnodes 2 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" /data/home/linjianma/metaseq/metaseq/launcher/opt_baselines.py --num-nodes 2 --num-gpus 8 -p test_v0 --model-size 175b --aws --benchmark #--local
# 8m --tensor-parallel-init-model-on-gpu --memory-efficient-fp16  --no-reshard-after-forward --tensor-parallel-init-model-on-gpu  --checkpoint-activations # change dummy_lm to streaming_language_modeling --dict-size 51196
# srun python metaseq_cli/train.py --distributed-world-size 16 --distributed-port 13413 --fp16 --save-dir results --cluster-env aws --task streaming_language_modeling --data /datasets01/gptz_corpus_dedup_10_10_1_0.05_exp29/120321/ --train-subset train --ignore-unused-valid-subsets --num-workers 8 --num-workers-valid 1 --validate-interval-updates 2000 --save-interval-updates 2000 --no-epoch-checkpoints --no-best-checkpoints --fp16-init-scale 4 --ddp-backend ptd_fully_sharded --use-sharded-state --model-parallel-size 1 --criterion vocab_parallel_cross_entropy --distribute-checkpointed-activations --full-megatron-init --megatron-init-sigma 0.006 --activation-fn relu --arch transformer_lm_megatron --share-decoder-input-output-embed --decoder-layers 4 --decoder-embed-dim 128 --decoder-ffn-embed-dim 512 --decoder-attention-heads 2 --decoder-learned-pos --no-scale-embedding --tokens-per-sample 2048 --optimizer adam --adam-betas '(0.9, 0.95)' --adam-eps 1e-08 --clip-norm 1.0 --clip-norm-type l2 --lr-scheduler polynomial_decay --lr 0.001 --end-learning-rate 0.0001 --warmup-updates 50 --total-num-update 50 --dropout 0.1 --attention-dropout 0.1 --no-emb-dropout --weight-decay 0.1 --batch-size 32 --update-freq 1 --max-update 50 --seed 1 --log-format json --log-interval 5 --required-batch-size-multiple 1
# 1.3b change batch size to 64 --tensor-parallel-init-model-on-gpu --memory-efficient-fp16  --no-reshard-after-forward
# srun python metaseq_cli/train.py --distributed-world-size 16 --distributed-port 15286 --fp16 --save-dir results --cluster-env aws --task streaming_language_modeling --dict-size 51196 --train-subset train --ignore-unused-valid-subsets --num-workers 8 --num-workers-valid 1 --validate-interval-updates 2000 --save-interval-updates 2000 --no-epoch-checkpoints --no-best-checkpoints  --fp16-init-scale 4 --ddp-backend ptd_fully_sharded --use-sharded-state --checkpoint-activations --model-parallel-size 2 --criterion vocab_parallel_cross_entropy --distribute-checkpointed-activations --full-megatron-init --megatron-init-sigma 0.006 --activation-fn relu --arch transformer_lm_megatron --share-decoder-input-output-embed --decoder-layers 24 --decoder-embed-dim 2048 --decoder-ffn-embed-dim 8192 --decoder-attention-heads 32 --decoder-learned-pos --no-scale-embedding --tokens-per-sample 2048 --optimizer adam --adam-betas '(0.9, 0.95)' --adam-eps 1e-08 --clip-norm 1.0 --clip-norm-type l2 --lr-scheduler polynomial_decay --lr 0.0002 --end-learning-rate 2e-05 --warmup-updates 50 --total-num-update 50 --dropout 0.1 --attention-dropout 0.1 --no-emb-dropout --weight-decay 0.1 --batch-size 16 --update-freq 1 --max-update 50 --seed 1 --log-format json --log-interval 5 --required-batch-size-multiple 1
# 175b # --no-reshard-after-forward  --tensor-parallel-init-model-on-gpu --memory-efficient-fp16
srun python metaseq_cli/train.py --distributed-world-size 128 --model-parallel-size 8 --batch-size 4 --distributed-port 11452 --fp16 --save-dir results --cluster-env aws --task dummy_lm --dict-size 51196 --train-subset train --ignore-unused-valid-subsets --num-workers 8 --num-workers-valid 1 --validate-interval-updates 2000 --save-interval-updates 2000 --no-epoch-checkpoints --no-best-checkpoints --fp16-init-scale 4 --ddp-backend ptd_fully_sharded --use-sharded-state --criterion vocab_parallel_cross_entropy --distribute-checkpointed-activations --full-megatron-init --megatron-init-sigma 0.006 --activation-fn relu --arch transformer_lm_megatron --share-decoder-input-output-embed --decoder-layers 96 --decoder-embed-dim 12288 --decoder-ffn-embed-dim 49152 --decoder-attention-heads 96 --decoder-learned-pos --no-scale-embedding --tokens-per-sample 2048 --optimizer adam --adam-betas '(0.9, 0.95)' --adam-eps 1e-08 --clip-norm 1.0 --clip-norm-type l2 --lr-scheduler polynomial_decay --lr 3e-05 --end-learning-rate 3e-06 --warmup-updates 50 --total-num-update 50 --dropout 0.1 --attention-dropout 0.1 --no-emb-dropout --weight-decay 0.1 --update-freq 1 --max-update 50 --seed 1 --log-format json --log-interval 5 --required-batch-size-multiple 1 --checkpoint-activations --use-non-recursive --bucket-size 900000000 #--backward-prefetch pre
