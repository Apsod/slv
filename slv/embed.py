from abc import *
import json

from transformers import AutoTokenizer, AutoModel
import torch
import itertools

from io import StringIO
from html.parser import HTMLParser



class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def extract(line):
    doc = json.loads(line)
    return doc['question']['id'], strip_tags(doc['question']['text'])

def chunkit(xs, batch_size):
    it = iter(xs)
    chunk = list(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, batch_size))


class Embedder(object):
    @abstractmethod
    def embed(self, texts):
        """
        texts: list of N texts
        returns: N x D matrix (pytorch?) 
        """
        pass

    def run_on(self, json_path, batch_size=16):
        chunks = []
        ixs = []
        for batch in chunkit(map(extract, open(json_path, 'rt')), batch_size):
            ids, txts = zip(*batch)
            ixs.extend(ids)
            chunks.append(self.embed(txts))
        vec = torch.cat(chunks, dim=0)
        return ixs, vec

    def run_and_save(self, json_path, out_path, batch_size=16):
        ixs, vec = self.run_on(json_path, batch_size=batch_size)
        torch.save({'ixs':ixs, 'vecs': vec}, out_path)


class CT_Embedder(Embedder):
    def __init__(self):
        model_path = 'kb-bert-ct'
        kb_path = 'KB/bert-base-swedish-cased'

        model = AutoModel.from_pretrained(model_path).cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        
        tokenizer = AutoTokenizer.from_pretrained(kb_path)
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, texts):
        batch = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                max_length=512,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt')
        ixs = batch['input_ids']
        att = batch['attention_mask']
        return ixs, att

    def embed(self, texts):
        ixs, att = self.encode(texts)
        ixs = ixs.to(device=self.model.device)
        att = att.to(device=self.model.device)

        outs, *_ = self.model(input_ids=ixs, attention_mask=att)
        mean_mask = torch.true_divide(att, att.sum(1, keepdims=True))
        means = (outs * mean_mask.unsqueeze(2)).sum(1)
        return means
    
if __name__ == '__main__':
    embedder = CT_Embedder()
    embedder.run_and_save('/home/amaru/git/slv/data/dump.json', 'ct_rep')
