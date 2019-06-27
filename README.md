# emotionX2019

This repository contains the code of team Antenna for EmotionX 2019 competition. 

## Data Loading
Data loading is a little bit complicated in this repository. Since we use [BERT](https://github.com/google-research/bert) for pre-feature extractoin of oru model, we need to first convert raw dialogues into the format that BERT can read. Thus, our data loading steps include: 
1. convert raw dialogues into the format. This is done by `./data/preprocess.py` and results are stored in corresponding folders in `./data/EmotionPush_bert` and `./data/Friends_bert`. Results of this step are already commited. 
2. Clone BERT and download pre-trained models. We use uncased BERT Base for this project.
3. Fine-tuning and extract features on BERT. Due to the limitation of computation resource, we have to use the source code to do fine-tuning to get feature vectors. [BERT](https://github.com/google-research/bert) contains a detailed instruction on how to do this but to save you some effort, I include a script `run_bert.sh` to do this. All you need to do is modifying the input and output directory as well as which dataset you want to run(Friends or EmotionPush). Specifically, `input_dir` of `create_pretraining_data.py` and `extract_features.py` should be the location of the bert format input data as described in step 1. The out_dir of `extract_features.py` is a customized path. The ones that I used can be found in `data_split.py` in input path variables. 
4. `./data/data_split.py` then reads the results from output directories in step 3 and do the train/development split. Training scripts would load them at runtime. 

## Training
The training stage is easy. Simply run `./run_train.sh`. Hyper parameters can be tuned in the script as well. Note that not all hyper parameters are included in the script. Some of them have to be tuned mannually in `model.py`. 
