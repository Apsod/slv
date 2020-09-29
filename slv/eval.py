from slv.util.tags import TAGS
from slv.query import load
import json
from collections import defaultdict

def eval(data_path, **kwargs):
    tagset = set(TAGS.keys())
    clusters = defaultdict(list)
    for doc in map(json.loads, open(data_path)):
        tags = set(doc['question']['extras']['tags']).intersection(tagset)
        for tag in tags:
            cluster[tag].append(doc['question']['id'])


