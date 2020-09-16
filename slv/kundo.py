import requests
import math
import os
import functools
import logging
from dataclasses import dataclass

KEY = os.environ.get('KUNDO')

KUNDO = 'https://kundo.se'

def mkpath(*parts):
    return '/'.join(parts)

def emptyGenerator():
    if False:
        yield None

class SizedGenerator(object):
    def __init__(self, length, f):
        self.consumed = False
        self.f = f
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        assert not self.consumed, "SizedGenerator can only be run once!"
        self.consumed = True
        return self.f()


class Kundo(object):
    def __init__(self, key=None, include_all=False):
        self.session = requests.Session()
        self.set_key(key)
        self.include_all(include_all)


    def set_key(self, key=None):
        if key is None:
            self.session.params.pop('key', False)
        else:
            self.session.params['key'] = key

    def include_all(self, flag=True):
        if flag:
            self.session.params['include'] = 'all'
        else:
            self.session.params.pop('include', False)


    def get_all(self, *path, start=0, limit=50, **params):
        url = mkpath(KUNDO, *path)
        response0 = self.session.get(url, params=dict(start=start, limit=limit, **params))

        if response0.ok:
            total = int(response0.headers['X-TotalResults'])
            pages = math.ceil(total / limit)
            def g():
                yield from response0.json()
                for page in range(1, pages):
                    response = self.session.get(url, params=dict(start=limit*page, limit=limit, **params))
                    yield from response.json()
            return SizedGenerator(total, g)
        else:
            return SizedGenerator(0, emptyGenerator)

    
    def get_taglist(self):
        url = mkpath(KUNDO, 'api', 'taglist', 'livsmedelsverket.json')
        return self.session.get(url).json()


    def get_tagged(self, tag, **params):
        return self.get_all('api', 'tag', 'livsmedelsverket', '{}.json'.format(tag), **params)
    

    def get_questions(self, **params):
        return self.get_all('api', 'livsmedelsverket.json', **params)


    def get_answers(self, dialog_id, **params):
        return self.get_all('api', 'comment', 'livsmedelsverket', '{}.json'.format(dialog_id), **params)


    def get_dialogs(self, **params):
        questions = self.get_questions(**params)
        def g():
            for question in questions:
                answers = list(self.get_answers(question['id']))
                yield dict(question=question, answers=answers)
        return SizedGenerator(len(questions), g)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()


if __name__ == '__main__':
    import json
    import tqdm
    from slv.util.tags import TAGS
    def f():
        yield 1
        yield 2
        yield 3

    xs = SizedGenerator(3, f)

    with Kundo(KEY, include_all=True) as kundo:
        with open('dump.json', 'wt') as handle:
                for pair in tqdm.tqdm(kundo.get_dialogs()):
                    handle.write(json.dumps(pair))
                    handle.write('\n')
