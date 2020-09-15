import requests
import math
import os
import functools
from dataclasses import dataclass

KEY = os.environ.get('KUNDO')

KUNDO = 'https://kundo.se'

def mkpath(*parts):
    return '/'.join(parts)

def pagination_wrapper(request, limit=50):
    response = request(start=0, limit=limit)
    pages = math.ceil(response.total / limit)
    
    yield from response.json
    for page in range(1, pages):
        response = request(start=page*limit, limit=limit)
        yield from response.json

@dataclass
class ListResponse:
    total: int
    json: list

    @staticmethod
    def from_response(response):
        return ListResponse(int(response.headers['X-TotalResults']), response.json())


class Kundo(object):
    def __init__(self, key, include_all=False):
        self.session = requests.Session()
        self.session.params['key'] = key
        self.include_all(include_all)

    def include_all(self, flag=True):
        if flag:
            self.session.params['include'] = 'all'
        else:
            self.session.params.pop('include', False)
    
    def get_taglist(self):
        url = mkpath(KUNDO, 'api', 'taglist', 'livsmedelsverket.json')
        return self.session.get(url).json()

    def get_tagged(self, 

    def get_questions(self, **params):
        url = mkpath(KUNDO, 'api', 'livsmedelsverket.json')
        response = self.session.get(url, params=params)
        return ListResponse.from_response(response)

    def get_answers(self, dialog_id, **params):
        url = mkpath(KUNDO, 'api', 'comment', 'livsmedelsverket', '{}.json'.format(dialog_id))
        response = self.session.get(url, params=params)
        return ListResponse.from_response(response)

    def get_dialogs(self, **params):
        questions = self.get_questions(**params)
        pairs = []
        for question in questions.json:
            get_answer = functools.partial(self.get_answers, question['id'])
            answers = list(pagination_wrapper(get_answer))
            pairs.append(dict(
                question=question,
                answers=answers
                ))

        return ListResponse(questions.total, pairs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()


if __name__ == '__main__':
    import json
    with Kundo(KEY) as kundo:
        response = kundo.get_taglist()
        print(json.dumps(response))
