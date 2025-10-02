#!/bin/bash
#SBATCH --job-name=gpt2_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00              #Request 26 hours (2 extra hours)
#SBATCH --mem=128GB                  #Request 128GB per node
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --gres=gpu:a100:1            #Request one A100 GPU to use

#SBATCH --output=gpt2_train.%j.log            #Redirect stdout/err to file

# Run the training script
# ./train_gpt2cu     -i "/scratch/group/csce689609/llm.c/dev/data/fineweb10B/fineweb_train_*.bin"     -j "/scratch/group/csce689609/llm.c/dev/data/fineweb10B/fineweb_val_*.bin"     -o log124M     -e "d12"     -b 32 -t 1024     -d 524288     -r 0     -z 1     -c 0.1     -l 0.0006     -q 0.0     -u 700     -n 5000     -v 250 -s 20000     -h 1

# python training
# python train_gpt2.py --input_bin "/scratch/group/csce689609/llm.c/dev/data/fineweb10B/fineweb_train_*.bin" --input_val_bin "/scratch/group/csce689609/llm.c/dev/data/fineweb10B/fineweb_val_*.bin" --output_dir log124M_python --model=d12 --batch_size=32 --sequence_length=1024 --total_batch_size=524288 --zero_stage=1 --weight_decay=0.1 --learning_rate=0.0006 --learning_rate_decay_frac=0.0 --warmup_iters=700 --sample_every=20000 --val_loss_every=250 -checkpoint_every=5000 

# python train_gpt2.py --input_bin "/home/grads/a/aayushgautam/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin" --input_val_bin "/home/grads/a/aayushgautam/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin" --output_dir log124M_python --model=d12 --batch_size=4 --sequence_length=1024 --total_batch_size=-1 --zero_stage=1 --weight_decay=0.1 --learning_rate=0.0006 --learning_rate_decay_frac=0.0 --warmup_iters=700 --sample_every=2 --val_loss_every=2 --checkpoint_every=2 