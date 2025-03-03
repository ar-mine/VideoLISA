# VideoLISA

### data_download.py
Download videos in *stage2.json* and save available labels into `train-{NUM}.json` and `test-{NUM}.json`.

Notes: This scripts can cause some bugs under Windows.

### evaluate.py

## Train
`
accelerate launch --config_file configs/accelerate/zero3.yaml --num_processes=4 \
videolisa/train.py \
--config configs/default.yaml \
--per_device_train_batch_size=2 --num_train_epochs=5
`