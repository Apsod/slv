import requests
import math
import os
import functools
from dataclasses import dataclass

KEY = os.environ.get('KUNDO')

KUNDO = 'https://kundo.se'

def mkpath(*parts):
    return '/'.join(parts)


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
        response = self.session.get(url, params=dict(start=start, limit=limit, **params))
        total = int(response.headers['X-TotalResults'])
        pages = math.ceil(total / limit)
        yield from response.json()
        for page in range(1, pages):
            response = self.session.get(url, params=dict(start=limit*page, limit=limit, **params))
            yield from response.json()

    
    def get_taglist(self):
        url = mkpath(KUNDO, 'api', 'taglist', 'livsmedelsverket.json')
        return self.session.get(url).json()


    def get_tagged(self, tag, **params):
        yield from self.get_all('api', 'tag', 'livsmedelsverket', '{}.json'.format(tag), **params)
    

    def get_questions(self, **params):
        yield from self.get_all('api', 'livsmedelsverket.json', **params)


    def get_answers(self, dialog_id, **params):
        yield from self.get_all('api', 'comment', 'livsmedelsverket', '{}.json'.format(dialog_id), **params)


    def get_dialogs(self, **params):
        questions = self.get_questions(**params)
        pairs = []
        for question in questions:
            answers = list(self.get_answers(question['id']))
            yield dict(question=question, answers=answers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()


if __name__ == '__main__':
    import json
    with Kundo(KEY, include_all=True) as kundo:
        for pair in kundo.get_dialogs(tags='a catering'):
            print(json.dumps(pair))
