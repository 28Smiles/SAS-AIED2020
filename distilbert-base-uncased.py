import os

# -- GPU TO USE --
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -- PARAMETERS --
MODEL_NAME = 'distilbert-base-uncased'
MODEL_PREFIX = 'DistilBert'
DATASET = 'union'
LANGS = ['en']
TRAIN_BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARN_RATE = 1e-5
EPOCHS = 24
WARMUP_STEPS = 1024
SEQUENCE_LENGTH = 512
# ----------------


import json
from transformers import *
from torch.utils.data import DataLoader, RandomSampler
from util.train import training
from util.dataset import load_semeval, tokenize, dataset
from util.val_datasets import val_datasets
from util.hotload import load_model
from itertools import chain

EXPERIMENT = '{}-{:d}-{:.0E}-{}-{}'.format(MODEL_NAME, TRAIN_BATCH_SIZE * ACCUMULATION_STEPS, LEARN_RATE, DATASET,
                                           '_'.join(LANGS))

# Create log and dump config
output_dir = 'logs/{}'.format(EXPERIMENT)
if os.path.exists(output_dir):
    raise RuntimeError('Experiment already runned!')
else:
    os.makedirs(output_dir)
    with open(output_dir + '/parameters.json', 'w+') as config_file:
        json.dump({
            'MODEL_PREFIX': MODEL_PREFIX,
            'MODEL_NAME': MODEL_NAME,
            'DATASET': DATASET,
            'LANGS': LANGS,
            'TRAIN_BATCH_SIZE': TRAIN_BATCH_SIZE,
            'ACCUMULATION_STEPS': ACCUMULATION_STEPS,
            'LEARN_RATE': LEARN_RATE,
            'EPOCHS': EPOCHS,
            'WARMUP_STEPS': WARMUP_STEPS,
            'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
        }, config_file, sort_keys=True, indent=4, separators=(',', ': '))

# Load and initialize model
MODEL_CLASS = load_model(MODEL_PREFIX)
TOKENIZER = MODEL_CLASS[0].from_pretrained(MODEL_NAME)
CONFIG = MODEL_CLASS[1].from_pretrained(MODEL_NAME, num_labels=3)
MODEL = MODEL_CLASS[2].from_pretrained(MODEL_NAME, config=CONFIG)

# Load training data
train_dataset = dataset(
    tokenize(chain(*(load_semeval(DATASET, 'train', lang) for lang in LANGS)), TOKENIZER, SEQUENCE_LENGTH))
train_sampler = RandomSampler(train_dataset)
train_dataset = DataLoader(train_dataset, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE, drop_last=True)

# Run Training
training(
  train_dataset,
  val_datasets(TOKENIZER, SEQUENCE_LENGTH),
  MODEL,
  EXPERIMENT,
  LEARN_RATE,
  WARMUP_STEPS,
  TRAIN_BATCH_SIZE,
  EPOCHS,
  ACCUMULATION_STEPS
)
