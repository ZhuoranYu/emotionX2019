import os
import json
import random

path = './EmotionPush'

with open(os.path.join(path, 'processed.json')) as fp:
    data = json.loads(fp.read())
n_samples = len(data)
n_train = int(n_samples * 0.8)
random.shuffle(data)


###########################################################
embedding_path = './EmotionPush_bert/en_embedding.json'
# data split
train_samples = data[:n_train]
develop_samples = data[n_train:]

# Load Bert Embeddings
embeddings_en = {}
tokens_en = {}
with open(embedding_path) as fp:
    for line in fp:
        entry = json.loads(line)
        idx = entry['linex_index']
        cls = entry['features'][0]
        assert cls['token'] == '[CLS]'
        values = cls['layers'][0]['values']
        embeddings_en[idx] = values

        tokenList = entry['features'][1:]
        tokens_en[idx] = tokenList

training_file = []
develop_file = []

for dialogue in train_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_en[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        tokens_utt = tokens_en[line_index]

        utt = {'uid': utterance['uid'], 'embedding': vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_en']}

        batch.append(utt)
    training_file.append(batch)

for dialogue in develop_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_en[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        tokens_utt = tokens_en[line_index]

        utt = {'uid': utterance['uid'], 'embedding':vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_en']}

        batch.append(utt)
    develop_file.append(batch)

with open(os.path.join(path, 'train', 'en_train.json'), 'w') as fp:
    json.dump(training_file, fp)

with open(os.path.join(path, 'develop', 'en_develop.json'), 'w') as fp:
    json.dump(develop_file, fp)

print("done")


embedding_path = './EmotionPush_bert/de_embedding.json'

# data split
train_samples = data[:n_train]
develop_samples = data[n_train:]

# Load Bert Embeddings
embeddings_de = {}
tokens_de = {}
with open(embedding_path) as fp:
    for line in fp:
        entry = json.loads(line)
        idx = entry['linex_index']
        cls = entry['features'][0]
        assert cls['token'] == '[CLS]'
        values = cls['layers'][0]['values']
        embeddings_de[idx] = values

        tokenList = entry['features'][1:]
        tokens_de[idx] = tokenList

training_file = []
develop_file = []

for dialogue in train_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_de[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        tokens_utt = tokens_de[line_index]

        utt = {'uid': utterance['uid'], 'embedding': vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_de']}

        batch.append(utt)
    training_file.append(batch)

for dialogue in develop_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_de[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        tokens_utt = tokens_de[line_index]

        utt = {'uid': utterance['uid'], 'embedding':vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_de']}

        batch.append(utt)
    develop_file.append(batch)

with open(os.path.join(path, 'train', 'de_train.json'), 'w') as fp:
    json.dump(training_file, fp)

with open(os.path.join(path, 'develop', 'de_develop.json'), 'w') as fp:
    json.dump(develop_file, fp)

print("done")

embedding_path = './EmotionPush_bert/fr_embedding.json'
train_samples = data[:n_train]
develop_samples = data[n_train:]
# Load Bert Embeddings
embeddings_fr = {}
tokens_fr = {}
with open(embedding_path) as fp:
    for line in fp:
        entry = json.loads(line)
        idx = entry['linex_index']
        cls = entry['features'][0]
        assert cls['token'] == '[CLS]'
        values = cls['layers'][0]['values']
        embeddings_fr[idx] = values

        tokenList = entry['features'][1:]
        tokens_fr[idx] = tokenList

training_file = []
develop_file = []

for dialogue in train_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_fr[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']

        tokens_utt = tokens_fr[line_index]

        utt = {'uid': utterance['uid'], 'embedding': vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_fr']}

        batch.append(utt)
    training_file.append(batch)

for dialogue in develop_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_fr[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        tokens_utt = tokens_fr[line_index]

        utt = {'uid': utterance['uid'], 'embedding':vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_fr']}

        batch.append(utt)
    develop_file.append(batch)

with open(os.path.join(path, 'train', 'fr_train.json'), 'w') as fp:
    json.dump(training_file, fp)

with open(os.path.join(path, 'develop', 'fr_develop.json'), 'w') as fp:
    json.dump(develop_file, fp)

print("done")

embedding_path = './EmotionPush_bert/it_embedding.json'

# data split
train_samples = data[:n_train]
develop_samples = data[n_train:]

# Load Bert Embeddings
embeddings_it = {}
tokens_it = {}
with open(embedding_path) as fp:
    for line in fp:
        entry = json.loads(line)
        idx = entry['linex_index']
        cls = entry['features'][0]
        assert cls['token'] == '[CLS]'
        values = cls['layers'][0]['values']
        embeddings_it[idx] = values

        tokenList = entry['features'][1:]
        tokens_it[idx] = tokenList
training_file = []
develop_file = []

for dialogue in train_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_it[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        tokens_utt = tokens_it[line_index]

        utt = {'uid': utterance['uid'], 'embedding': vector, 'tokens':tokens_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_it']}

        batch.append(utt)
    training_file.append(batch)

for dialogue in develop_samples:
    batch = []
    for utterance in dialogue:
        line_index = utterance['line_index']
        vector = embeddings_it[line_index]
        annotation = utterance['annotation']
        speaker = utterance['speaker']
        token_utt = tokens_it[line_index]

        utt = {'uid': utterance['uid'], 'embedding':vector, 'tokens':token_utt, 'annotation': annotation, 'speaker': speaker, 'utt':utterance['utt_it']}

        batch.append(utt)
    develop_file.append(batch)

with open(os.path.join(path, 'train', 'it_train.json'), 'w') as fp:
    json.dump(training_file, fp)

with open(os.path.join(path, 'develop', 'it_develop.json'), 'w') as fp:
    json.dump(develop_file, fp)

print("done")



