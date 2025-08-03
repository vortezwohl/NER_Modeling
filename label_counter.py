import json
from data_process import processed_labels

with open('./data/datasets.json', 'r', encoding='utf-8') as f:
    datasets = json.load(f)

train_dataset = datasets['train']
valid_dataset = datasets['valid']
test_dataset = datasets['test']


def count_labels(s: str) -> list:
    labels = []
    for label in processed_labels:
        labels.append((label, s.count(str(label) + ']')))
    return labels


print('TrainSet', count_labels(json.dumps(train_dataset, ensure_ascii=False)))
print('ValidSet', count_labels(json.dumps(valid_dataset, ensure_ascii=False)))
print('TestSet', count_labels(json.dumps(test_dataset, ensure_ascii=False)))
