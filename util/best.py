import os
import tensorflow as tf
import json
import torch
from util.hotload import load_model


def parse(file):
    o = {}
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if not v.tag == 'loss':
                if not e.step in o:
                    o[e.step] = {}

                o[e.step][v.tag] = v.simple_value

    return o


def sumup(values, parts):
    s = 0
    for part in parts:
        s += values[part]

    return s


def scores(values, parts):
    o = {}
    for k, v in values.items():
        o[k] = sumup(v, parts)

    return o


def max_key(values):
    k, v = 0, 0
    for key, value in values.items():
        k, v = key, value
        break

    for key, value in values.items():
        if value >= v:
            k, v = key, value

    return k, v


def log(experiment):
    for file in os.listdir('logs/' + experiment):
        if file.startswith('events.out.tfevents'):
            return 'logs/' + experiment + '/' + file


def load_parameters(experiment):
    with open('logs/{}/parameters.json'.format(experiment)) as json_file:
        return json.load(json_file)


def load_best(experiment, parts=['unseen_answers_en_accuracy_3_way', 'unseen_questions_en_accuracy_3_way',
                                 'unseen_domains_en_accuracy_3_way']):
    log_data = parse(log(experiment))
    best_step, _ = max_key(scores(log_data, parts))
    parameters = load_parameters(experiment)

    MODEL_CLASS = load_model(parameters['MODEL_PREFIX'])
    TOKENIZER = MODEL_CLASS[0].from_pretrained(parameters['MODEL_NAME'])
    CONFIG = MODEL_CLASS[1].from_pretrained(parameters['MODEL_NAME'], num_labels=3)
    MODEL = MODEL_CLASS[2].from_pretrained(parameters['MODEL_NAME'], config=CONFIG)
    MODEL.load_state_dict(torch.load('logs/{}/model_{}.torch'.format(experiment, best_step), map_location='cpu'))

    return log_data[best_step], MODEL, TOKENIZER
