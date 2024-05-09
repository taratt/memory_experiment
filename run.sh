module load StdEnv/2023 nvhpc/23.9
module load python/3.11.5
module load cuda/12.2

export TRANSFORMERS_OFFLINE=true
source ENV/bin/activate

for model_size in 125m 350m 1.3b 2.7b 6.7b 13b
do
    for lora_rank in 0 0.01 0.05 0.1 0.2
        do
		python3 mem_expr.py \
		--lora_rank $lora_rank \
        	--model facebook/opt-$model_size \
		--data_type fp16
    done
done
