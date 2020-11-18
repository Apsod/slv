"""
Main module, defining all slv entrypoints.
"""

import argparse
import json
import sys
import os

import tqdm

from slv.matcher.trainer import _train
from slv.matcher.embed import _embed
from slv.kundo import Kundo
from slv import TAG_SET

def is_train(doc):
    """
    Checks whether the document is tagged with one of the special tags, e.g. "o import animaliska", in which case it should not be part of the training set. 
    """
    try:
        tags = set(doc['question']['extras']['tags'])
        return tags.isdisjoint(TAG_SET)
    except KeyError:
        return True


def _get_split(sp):
    parser = sp.add_parser('split', help='Get ALL data from Kundo, split into train and test data-sets (test sets contains tagged documents')
    parser.add_argument(
            '--out_dest',
            type=str)

    parser.add_argument(
            '--kundo_key',
            type=str,
            default=os.environ.get('KUNDO')
    )

    def go(args):
        with \
                open('{}.train.json'.format(args.out_dest), 'xt') as train, \
                open('{}.test.json'.format(args.out_dest), 'xt') as test, \
                Kundo(args.kundo_key, include_all=True) as kundo:
            for doc in tqdm.tqdm(kundo.get_dialogs()):
                if is_train(doc):
                    train.write(json.dumps(doc))
                    train.write('\n')
                else:
                    test.write(json.dumps(doc))
                    test.write('\n')

    parser.set_defaults(go=go)

def _get_all(sp):
    parser = sp.add_parser('all', help='Get ALL data from Kundo')
    parser.add_argument(
            '--out',
            type=argparse.FileType('wt'),
            default=sys.stdout)

    parser.add_argument(
            '--kundo_key',
            type=str,
            default=os.environ.get('KUNDO')
    )

    def go(args):
        with Kundo(args.kundo_key, include_all=True) as kundo:
            for doc in tqdm.tqdm(kundo.get_dialogs()):
                args.out.write(json.dumps(doc))
                args.out.write('\n')

    parser.set_defaults(go=go)


def _get(sp):
    parser = sp.add_parser('get', help='commands for getting data from Kundo')
    subparsers = parser.add_subparsers()
    _get_all(subparsers)
    _get_split(subparsers)

def _model(sp):
    parser = sp.add_parser('model', help='commands for training and using a model')
    subparsers = parser.add_subparsers()
    _train(subparsers)
    _embed(subparsers)

def __main__():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    _get(subparsers)
    _model(subparsers)

    args = parser.parse_args()
    args.go(args)

