import os
import json

from tqdm import tqdm
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
max_other_token_count = 11_000
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
label_map = {
    0: 0,  # other
    1: 1,  # person
    3: 2,  # org
    5: 3,  # loc
    7: 4,  # misc
}
processed_labels = [0, 1, 2, 3, 4]


def token_decode(s: str) -> list:
    return [_.strip().replace("'", '').replace('[', '').replace(']', '') for _ in
            s.replace("' '", '||').replace('\n', '||').split('||')]


def label_decode(s: str) -> list:
    _labels = []
    label_lists = token_decode(s)
    for label_ls in label_lists:
        _labels.extend([int(_) for _ in label_ls.split(' ')])
    return _labels


def token_combination(T: list, L: list) -> tuple[list, list]:
    abandon_id = []
    for _i, _token in enumerate(T):
        if L[_i] > 0 and L[_i] % 2 == 0:
            if L[_i - 1] % 2 == 1 or L[_i - 1] == 0:
                T[_i - 1] = T[_i - 1] + ' ' + _token
                abandon_id.append(_i)
    ret_T = []
    ret_L = []
    for _i, _token in enumerate(T):
        if _i not in abandon_id:
            ret_T.append(_token)
            ret_L.append(L[_i])
    return ret_T, ret_L


def label_mapping(L: list) -> list:
    return [label_map[_] for _ in L]


def recursive_token_combination(T: list, L: list) -> tuple[list, list]:
    max_iter = 100
    _iter = 0
    while _iter < max_iter:
        _iter += 1
        T, L = token_combination(T, L)
        _continue = False
        for _l in L:
            if _l > 0 and _l % 2 == 0:
                _continue = True
        if _continue:
            continue
        return T, label_mapping(L)


for tokens, labels, lang in tqdm(total_train_data):
    tokens, labels = token_decode(tokens), label_decode(labels)
    tokens, labels = recursive_token_combination(tokens, labels)
    for i, token in enumerate(tokens):
        if token in train_tokens:
            continue
        train_tokens.add(token)
        if labels[i] > 0:
            train_dataset.append((token, labels[i]))
        else:
            if other_token_count_for_each_lang[lang] < max_other_token_count:
                other_token_count_for_each_lang[lang] += 1
                train_dataset.append((token, labels[i]))

for tokens, labels, lang in tqdm(total_valid_data):
    tokens, labels = token_decode(tokens), label_decode(labels)
    tokens, labels = recursive_token_combination(tokens, labels)
    for i, token in enumerate(tokens):
        if token in valid_tokens:
            continue
        valid_tokens.add(token)
        if labels[i] > 0:
            valid_dataset.append((token, labels[i]))
        else:
            if other_token_count_for_each_lang[lang] < max_other_token_count:
                other_token_count_for_each_lang[lang] += 1
                valid_dataset.append((token, labels[i]))

for tokens, labels, lang in tqdm(total_test_data):
    tokens, labels = token_decode(tokens), label_decode(labels)
    tokens, labels = recursive_token_combination(tokens, labels)
    for i, token in enumerate(tokens):
        if token in test_tokens:
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
