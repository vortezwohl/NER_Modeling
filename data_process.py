import os
import json

import pandas as pd

base_path = './data/multilingual_wikineural'
languages = ['de', 'en', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'ru']
train_data = dict()
valid_data = dict()
test_data = dict()

for lang in languages:
    train_data[lang] = pd.read_csv(os.path.join(base_path, f'train_{lang}.csv')).to_dict(orient='records')
    valid_data[lang] = pd.read_csv(os.path.join(base_path, f'val_{lang}.csv')).to_dict(orient='records')
    test_data[lang] = pd.read_csv(os.path.join(base_path, f'test_{lang}.csv')).to_dict(orient='records')

total_train_data = []
total_valid_data = []
total_test_data = []
for lang, data in train_data.items():
    for d in data:
        total_train_data.append((d['tokens'], d['ner_tags'], d['lang']))
for lang, data in valid_data.items():
    for d in data:
        total_valid_data.append((d['tokens'], d['ner_tags'], d['lang']))
for lang, data in test_data.items():
    for d in data:
        total_test_data.append((d['tokens'], d['ner_tags'], d['lang']))

train_tokens = set()
valid_tokens = set()
test_tokens = set()

train_dataset = []
valid_dataset = []
test_dataset = []
max_other_token_count = 5_000  # 55k
other_token_count_for_each_lang = {
    'de': 0,
    'en': 0,
    'es': 0,
    'fr': 0,
    'it': 0,
    'nl': 0,
    'pl': 0,
    'pt': 0,
    'ru': 0
}


def token_decode(s: str) -> list:
    return [_.strip().replace("'", '').replace('[', '').replace(']', '') for _ in
            s.replace("' '", '||').replace('\n', '||').split('||')]


def label_decode(s: str) -> list:
    _labels = []
    label_lists = token_decode(s)
    for label_ls in label_lists:
        _labels.extend([int(_) for _ in label_ls.split(' ')])
    return _labels


for tokens, labels, lang in total_train_data:
    tokens, labels = token_decode(tokens), label_decode(labels)
    for i, token in enumerate(tokens):
        if token in train_tokens:
            print(token, end=' ')
            continue
        train_tokens.add(token)
        if labels[i] > 0:
            train_dataset.append((token, labels[i]))
        else:
            if other_token_count_for_each_lang[lang] < max_other_token_count:
                other_token_count_for_each_lang[lang] += 1
                train_dataset.append((token, labels[i]))

for tokens, labels, lang in total_valid_data:
    tokens, labels = token_decode(tokens), label_decode(labels)
    for i, token in enumerate(tokens):
        if token in valid_tokens:
            print(token, end=' ')
            continue
        valid_tokens.add(token)
        if labels[i] > 0:
            valid_dataset.append((token, labels[i]))
        else:
            if other_token_count_for_each_lang[lang] < max_other_token_count:
                other_token_count_for_each_lang[lang] += 1
                valid_dataset.append((token, labels[i]))

for tokens, labels, lang in total_test_data:
    tokens, labels = token_decode(tokens), label_decode(labels)
    for i, token in enumerate(tokens):
        if token in test_tokens:
            print(token, end=' ')
            continue
        test_tokens.add(token)
        if labels[i] > 0:
            test_dataset.append((token, labels[i]))
        else:
            if other_token_count_for_each_lang[lang] < max_other_token_count:
                other_token_count_for_each_lang[lang] += 1
                test_dataset.append((token, labels[i]))

with open('./data/datasets.json', 'w', encoding='utf-8') as f:
    json.dump({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }, f)

print('Dataset initialized', list(zip(train_dataset[:23], train_dataset[:23])))
print(f'other_token_count_for_each_lang: {other_token_count_for_each_lang}')
print(f'{len(train_dataset)} tokens to train in total.')
