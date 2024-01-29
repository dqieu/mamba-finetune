# mamba-finetune

## Usage

```
python tokenize_dataset.py --jsonl_path data/train.jsonl --save_path data/train --max_seq_length 4096 --skip_overlength True
python train.py  --dataset_path data/train  --per_device_train_batch_size 1  --gradient_accumulation_steps 1  --num_train_epochs 1  --save_steps 1000  --save_total_limit 5  --learning_rate 1e-4  --save_safetensors false  --remove_unused_columns false  --dataloader_pin_memory false  --logging_steps 50  --output_dir output
```