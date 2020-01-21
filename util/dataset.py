from xml.dom import minidom
import torch
from torch.utils.data import TensorDataset


def load_semeval_meta(dataset, type, lang='en'):
    semeval_keys = {
        'correct': 2,
        'incorrect': 1,
        'contradictory': 0
    }

    file = minidom.parse('datasets/{}_{}/{}.xml'.format(dataset, lang, type))

    for e_id, exercise in enumerate(file.getElementsByTagName('exercise')):
        for r_id, reference in enumerate(exercise.getElementsByTagName('reference')):
            for a_id, answer in enumerate(exercise.getElementsByTagName('answer')):
                yield (
                    e_id,
                    r_id,
                    a_id,
                    reference.firstChild.data,
                    answer.firstChild.data,
                    semeval_keys[answer.attributes['accuracy'].value]
                )


def load_semeval(dataset, type, lang='en'):
    semeval_keys = {
        'correct': 2,
        'incorrect': 1,
        'contradictory': 0
    }

    file = minidom.parse('datasets/{}_{}/{}.xml'.format(dataset, lang, type))

    for exercise in file.getElementsByTagName('exercise'):
        for reference in exercise.getElementsByTagName('reference'):
            for answer in exercise.getElementsByTagName('answer'):
                yield (
                    reference.firstChild.data,
                    answer.firstChild.data,
                    semeval_keys[answer.attributes['accuracy'].value]
                )


def project_semeval(score):
    return {
        2: 1,
        1: 0,
        0: 0
    }[score]


def tokenize(loader, tokenizer, sequence_length=512):
    for r, a, l in loader:
        idx = tokenizer.encode(r, a, True)
        if len(idx) > sequence_length:
            continue

        mask = [1] * len(idx) + [0] * (sequence_length - len(idx))
        idx += [0] * (sequence_length - len(idx))

        yield idx, mask, l


def dataset(loader):
    features = list(loader)

    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.uint8)

    all_outputs = torch.tensor([f[2] for f in features], dtype=torch.uint8)

    return TensorDataset(all_input_ids, all_input_mask, all_outputs)
