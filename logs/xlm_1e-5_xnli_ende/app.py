import torch
from transformers import *
from flask import Flask, escape, request
from flask_cors import CORS

GPU_INFERENCE = True

MODEL_NAME = 'xlm-mlm-tlm-xnli15-1024'
SEQUENCE_LENGTH = 512
TOKENIZER = XLMTokenizer.from_pretrained(MODEL_NAME)
CONFIG = XLMConfig.from_pretrained(MODEL_NAME, num_labels = num_labels)
MODEL = XLMForSequenceClassification(config = CONFIG)
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
    l = request.args.get("lang")
    
    print("Evaluate: ", lang, r, a)
    
    idx = TOKENIZER.encode(r, a, True)
    if len(idx) > SEQUENCE_LENGTH:
        return "ERROR Sentence too long, please trunicate"
        
    mask = [1] * len(idx) + [0] * (SEQUENCE_LENGTH - len(idx))
    idx += [0] * (SEQUENCE_LENGTH - len(idx))
    
    mask = torch.tensor([ mask ]).long()
    idx = torch.tensor([ idx ]).long()
    lang = torch.tensor([ lang = [ MODEL.config.lang2id['de'] ] * SEQUENCE_LENGTH ]).long()
    
    if GPU_INFERENCE:
        mask = mask.cuda()
        idx = idx.cuda()
    
    with torch.no_grad():
        outputs = MODEL(
            idx, 
            attention_mask = mask,
            langs = lang
        )
    
    e = outputs[0][0].cpu().numpy()
    
    return {
      'contradictory': float(e[0]),
      'incorrect': float(e[1]),
      'correct': float(e[2])
    }
