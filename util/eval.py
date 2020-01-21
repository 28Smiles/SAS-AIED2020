import numpy as np
import pandas as pd
import torch
from .dataset import project_semeval
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, \
    matthews_corrcoef, r2_score, roc_auc_score


def evaluate(model, val_dataset):
    model.eval()
    scores = []

    y_true = []
    y_pred = []
    y_pred_raw = []

    for batch in val_dataset:
        with torch.no_grad():
            outputs = model(
                batch[0].long().to('cuda'),
                attention_mask=batch[1].long().to('cuda')
            )

        y_true.extend(batch[2].view(-1).numpy())
        y_pred.extend(outputs[0].to('cpu').float().argmax(1).numpy())
        y_pred_raw.extend(outputs[0].to('cpu').float().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_raw = np.array(y_pred_raw)

    scores.append((
        'precision_3_way',
        precision_score(y_true, y_pred.round(), labels=list(range(3)), average='weighted')
    ))
    scores.append((
        'recall_3_way',
        recall_score(y_true, y_pred.round(), labels=list(range(3)), average='weighted')
    ))
    scores.append((
        'f1_macro_3_way',
        f1_score(y_true, y_pred.round(), average='macro')
    ))
    scores.append((
        'f1_micro_3_way',
        f1_score(y_true, y_pred.round(), average='micro')
    ))
    scores.append((
        'matthews_3_way',
        matthews_corrcoef(y_true, y_pred.round())
    ))
    scores.append((
        'accuracy_3_way',
        accuracy_score(y_true, y_pred.round(), normalize=True)
    ))

    y_pred_2 = [project_semeval(v) for v in y_pred.round()]
    y_true_2 = [project_semeval(v) for v in y_true]

    scores.append((
        'precision_2_way_projected',
        precision_score(y_true_2, y_pred_2, labels=list(range(2)), average='weighted')
    ))
    scores.append((
        'recall_2_way_projected',
        recall_score(y_true_2, y_pred_2, labels=list(range(2)), average='weighted')
    ))
    scores.append((
        'f1_macro_2_way_projected',
        f1_score(y_true_2, y_pred_2, average='macro')
    ))
    scores.append((
        'f1_micro_2_way_projected',
        f1_score(y_true_2, y_pred_2, average='micro')
    ))
    scores.append((
        'matthews_2_way_projected',
        matthews_corrcoef(y_true_2, y_pred_2)
    ))
    scores.append((
        'accuracy_2_way_projected',
        accuracy_score(y_true_2, y_pred_2, normalize=True)
    ))

    return scores, (y_true, y_pred_raw)
