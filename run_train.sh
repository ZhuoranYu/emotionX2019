#!/bin/bash

python train.py \
	--input_dir=/nethome/zyu336/emotionX2019/data/Friends \
        --output_dir=/nethome/zyu336/emotionX2019/data/Friends \
        --epochs=20 \
        --batch_size=16 \
        --embedding_size=2048 \
	--optimizer=sgd \
        --lstm_dropout=0.5 \
        --cnn_dropout=0.5 \
        --lr=0.01 \
	--lr_decay=0.5 \
        --lstm_dim=512 \

