import json
from io import StringIO
from html.parser import HTMLParser

import torch

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

class KundoData(torch.utils.data.Dataset):
    def __init__(self, path, keep=lambda x: True):
        self.question = []
        self.answer = []
        with open(path, 'rt') as handle:
            for doc in filter(keep, map(json.loads, handle)):
                #title = strip_tags(doc['question']['title'])
                question = doc['question']['text']
                answer = doc['answers']
                if len(answer) == 1:
                    self.question.append(strip_tags(question))
                    self.answer.append(strip_tags(answer[0]['text']))
    
    def __len__(self):
        return len(self.question)

    def __getitem__(self, ix):
        return self.question[ix], self.answer[ix]

def mk_loader(tokenizer, pair_ds, max_len=512, batch_size=32, pin_memory=True, shuffle=True, num_workers=4):
        def encode(texts):
            batch = tokenizer.batch_encode_plus(
                    texts,
                    add_special_tokens=True,
                    max_length=max_len,
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt')

            return batch['input_ids'], batch['attention_mask']

        def collate(texts):
            questions, answers = zip(*texts)
            return encode(questions), encode(answers)

        return torch.utils.data.DataLoader(
                pair_ds,
                batch_size=batch_size,
                collate_fn=collate,
                shuffle=shuffle,
                drop_last=True,
                pin_memory=True,
                num_workers=True)

