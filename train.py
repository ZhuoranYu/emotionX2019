import os
import numpy as np
import argparse
import json
import random
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from model import emotionDetector
from loss import FocalLoss


annotation_order = ['neural', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust', 'non-neural']
label_order = ['neural', 'joy', 'sadness', 'anger', 'non-neural']
MAX_WORD_LEN = 15
MAX_SENT_LEN = 24

def evaluate(model, criterion, val_data):
    loss = 0
    correct_class = [0] * 8
    total_class = [0] * 8
    un_decided = 0
    for batch_idx, batch in enumerate(val_data):
        sequence, labels = batch

        if torch.cuda.is_available():
            sequence, labels = sequence.cuda(), labels.cuda()
        sequence, labels = Variable(sequence), Variable(labels)
        with torch.no_grad():
            output = model(sequence)

            loss += criterion(output, labels)
            _, pred = torch.max(output.data, 1)
            n_utt = output.shape[0]
        
            for i in range(0, n_utt):
                l = labels[i] # label of current utt
                
                if pred[i] == l:
                    correct_class[l] += 1
                total_class[l] += 1
    
    loss /= len(val_data)
    # compute unweighted accuracy
    wa = 0.0
    eval_idx = [0, 1, 2, 4]
    accList = []
    for idx in eval_idx:
        acc = correct_class[idx] / total_class[idx]
        accList.append(acc)
        wa += acc # accuracy for each class
    wa /= 4
    print(total_class[0], total_class[1], total_class[2], total_class[4])
    return loss, wa, accList

def get_label_vector(annotation):
    annotation = list(annotation)
    annotation = [int(l) for l in annotation]
    annotation = np.array(annotation)
    max_idx = np.argmax(annotation)
    annotation = annotation.tolist()
    if annotation.count(annotation[max_idx]) != 1:
        max_idx = 7
    return max_idx


def load_data(data_samples, data_type, batch_size):
    '''
    data_samples: train or develop
    data_type: indicator of training or developing
    return: pytorch data loader
    '''
    bert_data = []
    min_utt = 10000
    max_utt = 0
    avg_utt = 0
    count_utt = 0
    for dialogue in data_samples:
        batch_x = None
        batch_y = None
        n_utt = len(dialogue)
        if n_utt < min_utt:
            min_utt = n_utt
        if n_utt > max_utt:
            max_utt = n_utt 
        count_utt += 1
        avg_utt += n_utt
        for utterance in dialogue:
            utt_x = None
            vector = np.array(utterance['embedding']).reshape(1, 768)

            # sequence embedding
            tokens = utterance['tokens']
            n_tok = 0
            
            for tok in tokens:
                if n_tok >= MAX_WORD_LEN:
                    break
                value = tok['layers'][0]['values']
                if utt_x is None:
                    utt_x = np.array(value).reshape(1, 768)
                else:
                    curr_utt = np.array(value).reshape(1, 768)
                    utt_x = np.vstack((utt_x, curr_utt))
                n_tok += 1
            if n_tok < MAX_WORD_LEN:
                padding = np.zeros((MAX_WORD_LEN - n_tok, 768))
                utt_x = np.vstack((utt_x, padding))
            
            # create weighted label
            annotation = utterance['annotation']
            label = get_label_vector(annotation) # weighted vector label

            # stack CLS and words
            utt_x = np.vstack((vector, utt_x))

            utt_x = utt_x.reshape((1, 1, -1, 768)) # (n_utt, channel, n_word, dim_size)
            if batch_x is None:
                batch_x = utt_x
            else:
                batch_x = np.concatenate((batch_x, utt_x), axis=0)
            if batch_y is None:
                batch_y = np.array(label)
            else:
                batch_y = np.hstack((batch_y, label))
         
        batch_x = torch.tensor(batch_x, dtype=torch.float32)
        batch_y = torch.tensor(batch_y)
        bert_data.append((batch_x, batch_y))
                
    return bert_data

def data_augmentation(train_data, train_de, train_fr, train_it):
    augmented_data = []
    
    for batch_idx, batch in enumerate(train_data):
        dice = random.randint(0, 3)
        if dice == 0:
            augmented_data.append(batch)
        elif dice == 1:
            augmented_data.append(train_de[batch_idx])
        elif dice == 2:
            augmented_data.append(train_fr[batch_idx])
        elif dice == 3:
            augmented_data.append(train_it[batch_idx])
    random.shuffle(augmented_data)
    return augmented_data


def train(model, criterion, train_data, develop_data, optimizer, scheduler, num_epochs, train_de, train_fr, train_it, log_interval=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    examples_this_epoch = 0
    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        augmented_train = data_augmentation(train_data, train_de, train_fr, train_it)
        for batch_idx, batch in enumerate(augmented_train):
            sequence, labels = batch
            model.train(True)

            if use_gpu:
                sequence = sequence.cuda()
                labels = labels.cuda()
            sequence = Variable(sequence)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(sequence)

            loss = criterion(outputs, labels)
            
            loss.backward(retain_graph=True)
            optimizer.step()
 
            examples_this_epoch += sequence.shape[0]
            if batch_idx % log_interval == 0:
                val_loss, val_acc, accList = evaluate(model, criterion, develop_data)
                train_loss = loss.data
                epoch_progress = 100. * batch_idx / len(train_data)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Train Loss: {:.6f}\tVal Loss: {:0.6f}\tVal Unweighted Acc: {}'.format(
                    epoch, examples_this_epoch, len(train_data), 
                    epoch_progress, train_loss, val_loss, val_acc))
                print("Validation accuracy for each class: %f %f %f %f" % (accList[0], 
                    accList[1], accList[2], accList[3]))
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model_wts = model.state_dict()
                    location = (epoch, batch_idx)
    print('Best val Acc: {:.4f} obtained at epoch: {}, batch: {}'.format(best_acc, location[0], location[1]))
    model.load_state_dict(best_model_wts)
    return model
                


parser = argparse.ArgumentParser()

# output paths
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str, default="model-checkpoint")
parser.add_argument("--model_name", type=str, default="model.pickle")

# hyper parameters
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--embedding_size", type=int, default=768)
parser.add_argument("--lstm_dropout", type=float, default=0.4)
parser.add_argument("--cnn_dropout", type=float, default=0.4)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.5)
parser.add_argument("--minlr", type=float, default=2e-5)

parser.add_argument("--lstm_dim", type=int, default=256)
parser.add_argument("--encoder_dim", type=int, default=1)
parser.add_argument("--fc_dim", type=int, default=1)
parser.add_argument("--max_dia_len", type=int, default=32)
parser.add_argument("--display", type=int, default=100)

# focal loss
parser.add_argument("--focal_gamma", type=float, default=2)
#parser.add_argument("--focal_alpha", type=float, default=0.25)

focal_alpha = [0.1, 0.15, 0.6, 0, 0.15, 0, 0, 0]


params = parser.parse_args()
if not os.path.exists(params.output_dir):
    os.makedirs(params.output_dir)

# Load Training Data
with open(os.path.join(params.input_dir, 'train', 'en_train.json')) as fp:
    training_samples = json.loads(fp.read())
with open(os.path.join(params.input_dir, 'develop', 'en_develop.json')) as fp:
    developing_samples = json.loads(fp.read())

# Raw Training Data
training_data = load_data(training_samples, 'training', params.batch_size)
developing_data = load_data(developing_samples, 'developing', params.batch_size)

#data augmentation
with open(os.path.join(params.input_dir, 'train', "de_train.json")) as fp:
    training_de_samples = json.loads(fp.read())
with open(os.path.join(params.input_dir, 'train', 'fr_train.json')) as fp:
    training_fr_samples = json.loads((fp.read()))
with open(os.path.join(params.input_dir, 'train', 'it_train.json')) as fp:
    training_it_samples = json.loads((fp.read()))
training_data_de = load_data(training_de_samples, 'training', params.batch_size)
training_data_fr = load_data(training_fr_samples, 'training', params.batch_size)
training_data_it = load_data(training_it_samples, 'training', params.batch_size)


# Augmented Data
# with open(os.path.join(params.input_dir, 'train', 'de_train.json')) as fp:
#     de_augment = json.loads(fp.read())
# with open(os.path.join(params.input_dir, 'train', 'fr_train.json')) as fp:
#     fr_augment = json.loads(fp.read())
# with open(os.path.join(params.input_dir, 'train', 'it_train.json')) as fp:
#     it_augment = json.loads(fp.read())

#batch_train = create_batch(training_data, params.batch_size)
#batch_dev = create_batch(developing_data, params.batch_size)


model = emotionDetector(params.lstm_dim, len(annotation_order), params.lstm_dropout, params.cnn_dropout, params.batch_size) 

if params.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=params.lr)
elif params.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=params.lr_decay)

#criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = FocalLoss(params.focal_gamma, focal_alpha)
model = train(model, criterion, training_data, developing_data, optimizer, scheduler, 
        params.epochs, training_data_de, training_data_fr, training_data_it, log_interval=params.display)

torch.save(model, './friends.pth')


if __name__ == "__main__":
    pass

