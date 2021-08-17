# ReCoRD training/test code with pre-trained language model

How to Run?
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --lm_type electra-base --train_batch_size 8 --num_train_epochs 2 --output_dir ./save/tmp
```

You can try other language models by changing lm_type to electra, bert, roberta.

If there is no pickle file (cache for dataset preprocessing), please add --read_data flag.

You can change the number of GPU you use by changing the number of CUDA_VISIBLE_DEVICES=#.