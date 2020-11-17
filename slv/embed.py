from abc import *
import json

from slv.matcher.model import QA_model
from transformers import AutoTokenizer, AutoModel
import torch
import itertools

from io import StringIO

import gensim
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

class TFIDF_Embedder(Embedder):
    def __init__(self, source):
        data = []
        for doc in map(extract, open(source, 'rt')):
            data.append(gensim.utils.simple_preprocess(doc[1]))
        self.dct = gensim.corpora.Dictionary(data)
        corpus = [self.dct.doc2bow(line) for line in data]
        self.model = gensim.models.TfidfModel(corpus)


    def encode(self, texts):
        data = list(map(gensim.utils.simple_preprocess, texts))
        return [self.dct.doc2bow(line) for line in data]

    def embed(self, texts):
        bows = self.encode(texts)
        vecs = []
        for bow in bows:
            tfidf = self.model[bow]
            ixs, vals = zip(*tfidf) if tfidf else ([], [])
            ixs = torch.LongTensor([ixs])
            vals = torch.FloatTensor(vals)
            vecs.append(torch.sparse.FloatTensor(ixs, vals, (len(self.dct),)))
        return torch.stack(vecs)


class QA_Embedder(Embedder):
    def __init__(self, model_path):
        kb_path = 'KB/bert-base-swedish-cased'

        model, tokenizer = QA_model.init(kb_path)
        model.load_state_dict(torch.load(model_path))

        self.model = model.eval()
        self.tokenizer = tokenizer

        for p in self.model.parameters():
            p.requires_grad_(False)
            self.device = p.device
        
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
        ixs = ixs.to(device=self.device)
        att = att.to(device=self.device)

        return self.model.generate_left((ixs, att))


class CT_Embedder(Embedder):
    def __init__(self, model_path):
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
    DATAPATH = '/raid/amaru/slv.ndjson'
    #embedder = QA_Embedder('/home/amaru/git/slv/slv/matcher/qamodel.pkl')
    #embedder.run_and_save(DATAPATH, 'matcher_rep')
    
    embedder = TFIDF_Embedder(DATAPATH)
    embedder.run_and_save(DATAPATH, 'tfidf_rep')

    import pickle

    with open('tfidf_model', 'wb') as handle:
        pickle.dump(embedder, handle)
