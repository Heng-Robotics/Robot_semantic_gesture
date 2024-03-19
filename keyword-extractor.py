from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import torch
import argparse
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='BERT Keyword Extractor')
parser.add_argument('--sentence', type=str, default=' ',
                    help='sentence to get keywords')
parser.add_argument('--path', type=str, default='model.pt',
                    help='path to load model')
args = parser.parse_args()

tag2idx = {'F': 0, 'S': 1, 'T': 2,'O': 3}
tags_vals = ['F', 'S', 'T', 'O']

# start1 = time.time()
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
model = BertForTokenClassification.from_pretrained("./bert-base-uncased", num_labels=len(tag2idx))
# end1 = time.time()
# print("duration1", str(end1-start1))       

def keywordextract(sentence, path):
    text = sentence
    tkns = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model = torch.load(path)
    model.eval()
    prediction = []
    logit = model(tokens_tensor, token_type_ids=None,
                                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, axis=2)])
    for k, j in enumerate(prediction[0]):
        if j==1 or j==0:
            print(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k], j)
   

import pandas as pd
df = pd.read_csv("joke_test.csv")
sentence_j = df["joke"].tolist()
for sentence in sentence_j:
    start2 = time.time()
    keywordextract(sentence, args.path)
    end2 = time.time()
    print("duration2", str(end2-start2))    
