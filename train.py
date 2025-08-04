import json
import time
from random import shuffle
from random import randint

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from vortezwohl.cache import LRUCache
from deeplotx.util import sha256
from deeplotx import SoftmaxRegression, LongformerEncoder

from util import NUM_CLASSES, one_hot

lf_encoder = LongformerEncoder(model_name_or_path='severinsimmler/xlm-roberta-longformer-base-16384')
CACHE = LRUCache(capacity=16384)


def encode(text: str) -> torch.Tensor:
    key = sha256(text)
    if key in CACHE:
        return CACHE[key]
    emb = lf_encoder.encode(text, cls_only=False).mean(dim=-2, dtype=model.dtype)
    CACHE[key] = emb
    return emb


with open('./data/datasets.json', 'r', encoding='utf-8') as f:
    datasets = json.load(f)

train_dataset = datasets['train']
valid_dataset = datasets['valid']
test_dataset = datasets['test']
print('Dataset loaded')

model = SoftmaxRegression(input_dim=768, output_dim=NUM_CLASSES, num_heads=4, num_layers=3, expansion_factor=1.25,
                          bias=True, dropout_rate=0.2, head_layers=2)
print(model)

train_step = 0
valid_step = 0
writer = SummaryWriter()

acc_train_loss = 0.
acc_valid_loss = 0.
eval_interval = 2000
log_interval = 200
valid_log_interval = 50

shuffle(train_dataset)
shuffle(valid_dataset)
shuffle(test_dataset)
print('Dataset shuffled', list(train_dataset[:23]))

elastic_net_param = {
    'alpha': 2e-4,
    'rho': 0.2
}
learning_rate = 2e-6
num_epochs = 1500
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

try:
    for epoch in range(num_epochs):
        model.train()
        for i, (_token, _label) in enumerate(train_dataset):
            if _label == 4:
                continue
            _one_hot_label = one_hot(_label).to(model.dtype).to(model.device)
            outputs = model.forward(encode(_token))
            loss = loss_function(outputs, _one_hot_label) + model.elastic_net(alpha=elastic_net_param['alpha'],
                                                                              rho=elastic_net_param['rho'])
            acc_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_step % log_interval == 0 and train_step > 0:
                writer.add_scalar('train/loss', acc_train_loss / log_interval, train_step)
                print(f'- Train Step {train_step} Loss {acc_train_loss / log_interval} \\'
                      f'\nToken={_token}'
                      f'\nPred={outputs.tolist()}'
                      f'\nLabel={_one_hot_label.tolist()}', flush=True)
                acc_train_loss = 0.
            train_step += 1
            if train_step % eval_interval == 0:
                model.eval()
                rand_idx = randint(0, len(valid_dataset) - 501)
                with torch.no_grad():
                    for _i, (__token, __label) in enumerate(valid_dataset[rand_idx: rand_idx + 500]):
                        if __label == 4:
                            continue
                        _one_hot_label = one_hot(__label).to(model.dtype).to(model.device)
                        outputs = model.forward(encode(__token))
                        loss = loss_function(outputs, _one_hot_label)
                        acc_valid_loss += loss.item()
                        if valid_step % valid_log_interval == 0 and valid_step > 0:
                            writer.add_scalar('valid/loss', acc_valid_loss / valid_log_interval, valid_step)
                            print(f'- Valid Step {valid_step} Loss {acc_valid_loss / valid_log_interval} \\'
                                  f'\nToken={__token}'
                                  f'\nPred={outputs.tolist()}'
                                  f'\nLabel={_one_hot_label.tolist()}', flush=True)
                            acc_valid_loss = 0.
                        valid_step += 1
                model.train()
except KeyboardInterrupt:
    print(model.save(model_name=f'{time.time()}', model_dir='checkpoints'))
