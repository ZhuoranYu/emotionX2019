#!/bin/bash

python train.py \
	--input_dir=/nethome/zyu336/emotionX2019/data/Friends \
        --output_dir=/nethome/zyu336/emotionX2019/data/EmotionPush \
        --epochs=50 \
        --batch_size=20 \
        --embedding_size=2048 \
	--optimizer=sgd \
        --lstm_dropout=0.4 \
        --cnn_dropout=0.5 \
        --lr=0.05 \
	--lr_decay=0.9 \
        --lstm_dim=128 \
	--focal_gamma=0.5 \

