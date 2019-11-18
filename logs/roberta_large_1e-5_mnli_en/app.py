import torch
from transformers import *
from flask import Flask, escape, request
from flask_cors import CORS

GPU_INFERENCE = True

MODEL_NAME = 'roberta-large-mnli'
SEQUENCE_LENGTH = 512
TOKENIZER = RobertaTokenizer.from_pretrained(MODEL_NAME)
CONFIG = RobertaConfig.from_pretrained(MODEL_NAME, num_labels = 3)
MODEL = RobertaForSequenceClassification(CONFIG)
MODEL.load_state_dict(torch.load('model.torch', map_location='cpu'))
MODEL.eval()

if GPU_INFERENCE:
    MODEL = MODEL.cuda()

app = Flask(__name__)
CORS(app)

@app.route("/")
def evaluate():
    r = request.args.get("reference")
    a = request.args.get("answer")
    
    print("Evaluate: ", r, a)
    
    idx = TOKENIZER.encode(r, a, True)
    if len(idx) > SEQUENCE_LENGTH:
        return "ERROR Sentence too long, please trunicate"
        
    mask = [1] * len(idx) + [0] * (SEQUENCE_LENGTH - len(idx))
    idx += [0] * (SEQUENCE_LENGTH - len(idx))
    
    mask = torch.tensor([ mask ]).long()
    idx = torch.tensor([ idx ]).long()
    
    if GPU_INFERENCE:
        mask = mask.cuda()
        idx = idx.cuda()
    
    with torch.no_grad():
        outputs = MODEL(
            idx, 
            attention_mask = mask
        )
    
    e = outputs[0][0].cpu().numpy()
    
    return {
      'contradictory': float(e[0]),
      'incorrect': float(e[1]),
      'correct': float(e[2])
    }
