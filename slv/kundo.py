import requests
import math
import os
import functools
import logging


KUNDO = 'https://kundo.se'

def mkpath(*parts):
    return '/'.join(parts)

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

    @staticmethod
    def empty():

        def generator():
            if False:
                yield None

        return SizedGenerator(0, generator)


class Kundo(object):
    """
    A class dedicated to getting data from the Kundo API. 
    Can be used with context managers, e.g. 
    ```
    with Kundo(include_all=True) as kundo:
      for question kundo.get_questions():
        ...
    ```
    """
    def __init__(self, key=None, include_all=False):
        self.session = requests.Session()
        self.set_key(key)
        self.include_all(include_all)


    def set_key(self, key=None):
        """
        Set the API key (or remove it if key=None)
        """
        if key is None:
            self.session.params.pop('key', False)
        else:
            self.session.params['key'] = key

    def include_all(self, flag=True):
        """
        Set the flag to include all, (includes archived posts)
        """
        if flag:
            self.session.params['include'] = 'all'
        else:
            self.session.params.pop('include', False)


    def get_all(self, *path, start=0, limit=50, **params):
        """
        Helper method.
        Calls kundo at "path" and returns a sized generator (handling pagination et.c)
        For this to work, the path must be a path that returns X-TotalResults.
        (I think this should work for all Kundo API endpoint that returns lists)
        """
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
            return SizedGenerator.empty()
    

    def get_taglist(self):
        """
        Get all tags that are used by livsmedelsverket
        """
        url = mkpath(KUNDO, 'api', 'taglist', 'livsmedelsverket.json')
        return self.session.get(url).json()


    def get_tagged(self, tag, **params):
        """
        Get all documents with the specified tag
        """
        return self.get_all('api', 'tag', 'livsmedelsverket', '{}.json'.format(tag), **params)
    

    def get_questions(self, **params):
        """
        Get all questions
        """
        return self.get_all('api', 'livsmedelsverket.json', **params)


    def get_answers(self, dialog_id, **params):
        """
        Get all answers to a question specified by dialog_id
        """
        return self.get_all('api', 'comment', 'livsmedelsverket', '{}.json'.format(dialog_id), **params)


    def get_dialogs(self, **params):
        """
        Get all dialogs
        """
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
