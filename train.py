import os
import numpy as np
import argparse
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from model import emotionDetector
from model import x_entropy_loss

annotation_order = ['neural', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
MAX_WORD_LEN = 15


def evaluate(model, criterion, val_data):
    loss = 0
    correct_class = [0] * 7
    total_class = [0] * 7
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
            n_utt = labels.shape[0]
        
            for i in range(0, n_utt):
                l = labels[i, :] # label of current utt
                l_list = l.tolist()

                # skip padding
                if sum(l_list) == 0:
                    continue

                for weight in l:
                    if l_list.count(weight) > 1 and weight != 0: # if this weight is not unique, skip
                        # non-neural
                        un_decided += 1
                    else:
                        l = l.view(1, -1)
                        _, true_label = torch.max(l.data, 1)
                        if pred[i] == true_label:
                            correct_class[true_label] += 1
                        total_class[true_label] += 1
    
    loss /= len(val_data)
    # compute unweighted accuracy
    wa = 0.0
    for idx, score in enumerate(correct_class):
        wa += score / total_class[idx]
    wa /= len(total_class)
    return loss, wa


def get_label_vector(annotation):
    annotation = list(annotation)
    annotation = [int(l) for l in annotation]
    annotation = np.array(annotation)
    label = annotation / np.sum(annotation)
    return label


def load_data(data_samples, data_type):
    '''
    data_samples: train or develop
    data_type: indicator of training or developing
    return: pytorch data loader
    '''
    bert_data = []
    for dialogue in data_samples:
        batch_x = None
        batch_y = None
        for utterance in dialogue:
            utt_x = None
            vector = np.array(utterance['embedding']).reshape(1, 768)

            # sequence embedding
            tokens = utterance['tokens']
            n_tok = 0
            for tok in tokens:
                if n_tok >= MAX_WORD_LEN:
                    break
                if utt_x is None:
                    utt_x = np.array(tok['layers']).reshape(1, 768)
                else:
                    curr_utt = np.array(tok['layers']).reshape(1, 768)
                    utt_x = np.vstack((utt_x, curr_utt))
                n_tok += 1
            # create weighted label
            annotation = utterance['annotation']
            label = get_label_vector(annotation) # weighted vector label

            # stack CLS and words
            utt_x = np.vstack((vector, utt_x))
            if batch_x is None:
                batch_x = utt_x
                batch_y = label
            else:
                batch_x = np.vstack((batch_x, utt_x))
                batch_y = np.vstack((batch_y, label))
        batch_x = torch.tensor(batch_x, dtype=torch.float32)
        batch_y = torch.tensor(batch_y, dtype=torch.float32)
        bert_data.append((batch_x, batch_y))


    return bert_data


def train(model, criterion, train_data, develop_data, optimizer, scheduler, num_epochs, log_interval=100):
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

        for batch_idx, batch in enumerate(train_data):
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
                val_loss, val_acc = evaluate(model, criterion, develop_data)
                train_loss = loss.data
                epoch_progress = 100. * batch_idx / len(train_data)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Train Loss: {:.6f}\tVal Loss: {:0.6f}\tVal Unweighted Acc: {}'.format(
                    epoch, examples_this_epoch, len(train_data), 
                    epoch_progress, train_loss, val_loss, val_acc))
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

params = parser.parse_args()
if not os.path.exists(params.output_dir):
    os.makedirs(params.output_dir)

# Load Training Data
with open(os.path.join(params.input_dir, 'train', 'en_train.json')) as fp:
    training_samples = json.loads(fp.read())
with open(os.path.join(params.input_dir, 'develop', 'en_develop.json')) as fp:
    developing_samples = json.loads(fp.read())

# Raw Training Data
training_data = load_data(training_samples, 'training')
developing_data = load_data(developing_samples, 'developing')

# Augmented Data
# with open(os.path.join(params.input_dir, 'train', 'de_train.json')) as fp:
#     de_augment = json.loads(fp.read())
# with open(os.path.join(params.input_dir, 'train', 'fr_train.json')) as fp:
#     fr_augment = json.loads(fp.read())
# with open(os.path.join(params.input_dir, 'train', 'it_train.json')) as fp:
#     it_augment = json.loads(fp.read())



model = emotionDetector(params.lstm_dim, len(annotation_order), params.lstm_dropout, params.cnn_dropout) 



if params.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=params.lr)
elif params.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=params.lr_decay)
criterion = x_entropy_loss()
model = train(model, criterion, training_data, developing_data, optimizer, scheduler, params.epochs, log_interval=params.display)


if __name__ == "__main__":
    pass

