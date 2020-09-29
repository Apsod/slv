import torch
import json
import collections


def load(rep_path, txt_path):
    d = torch.load(rep_path)

    vecs = d['vecs']

    # v_ixs: [doc_id]
    v_ixs = d['ixs']

    # dix2vix
    dix2vix = {dix: vix for vix, dix in enumerate(v_ixs)}

    print(dix2vix)

    # docs = {doc_id: {vix: int, text: str, tags: [str]}}
    docs = collections.defaultdict(dict)

    for line in open(txt_path):
        doc = json.loads(line)
        dix = doc['question']['id']
        docs[dix]['text'] = doc['question']['text']
        docs[dix]['tags'] = doc['question']['extras']['tags']
        docs[dix]['vix'] = dix2vix[dix]

    return vecs, v_ixs, docs


def prompt_loop(rep_path, txt_path):
    vectors, v_ixs, docs = load(rep_path, txt_path)

    prompt = int(input('Enter text id: '))
    while prompt:
        print(docs[prompt]['text'])
        vix = docs[prompt]['vix']

        q_tags = set(docs[prompt]['tags'])

        v = vectors[vix]
        dots = vectors @ v
        norms = vectors.pow(2).sum(1).sqrt()
        sims = dots / (norms[vix] * norms)
        vals, nbh_ixs = sims.topk(10)
        for val, nbh_ix in zip(vals, nbh_ixs):
            r_tags = set(docs[v_ixs[nbh_ix]]['tags'])

            q_not_r = q_tags - r_tags
            r_not_q = r_tags - q_tags
            shared = q_tags.intersection(r_tags)

            print('{:.2f}  [{} ({}) {}]'.format(val, q_not_r, shared, r_not_q))
            print(docs[v_ixs[nbh_ix]]['text'])

        prompt = int(input('Enter text id: '))

if __name__ == '__main__':
    prompt_loop('ct_rep', '../data/dump.json')
