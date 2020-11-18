import json
import copy
import argparse

import torch
import streamlit as st

from slv.matcher.model import QA_model

parser = argparse.ArgumentParser('streamlit app')
parser.add_argument('--model_path', required=True)
parser.add_argument('--embeddings', required=True)

args = parser.parse_args()


@st.cache
def get_model():
    model, tokenizer = QA_model.load(args.model_path)
    model = model.eval()

    for p in model.parameters():
        p.requires_grad=False

    return model, tokenizer

@st.cache(allow_output_mutation=True)
def get_embeddings():
    answers = []
    a_m = []

    questions = []
    q_m = []

    with open(args.embeddings, 'rt') as handle:
        for doc in map(json.loads, handle):
            answers.append(doc['answer']['text'])
            a_m.append(doc['answer']['embeddings'])

            questions.append(doc['question']['text'])
            q_m.append(doc['question']['embedding'])

    a_m = torch.tensor(a_m)
    q_m = torch.tensor(q_m)

    return a_m, tuple(answers), q_m, tuple(questions)

def cos_sim(q, M):
    qn = q.pow(2).sum()
    Mn = M.pow(2).sum(1)

    return (M @ q) / (qn * Mn).sqrt()

def sm_sim(q, M):
    return torch.softmax(M @ q, 0)

def get_topk(scores, items, k):
    vals, ixs = zip(*[(val.item(), items[ix]) for val, ix in zip(*scores.topk(k))])
    return {'score': vals, 'text': ixs}


def main():
    model, tokenizer = get_model()
    A, answers, Q, questions = get_embeddings()

    which = st.sidebar.selectbox('Vill du söka på en fråga eller ett svar?', ('Fråga', 'Svar'))

    K = st.sidebar.number_input('Antal grannar att visa', min_value=1, value=10)
    
    st.title('SLV: Frågor och Svar')

    query = st.text_input('Skriv {}'.format('en fråga' if which=='Fråga' else 'ett svar'))

    if query:
        batch = tokenizer.batch_encode_plus(
                [query],
                add_special_tokens=True,
                max_length=512,
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt')

        ixs, att = batch['input_ids'], batch['attention_mask']
        if which == 'Fråga':
            vec = model.generate_left((ixs, att))[0]
            left, right = st.beta_columns(2)
            with left:
                st.header('Liknande frågor')
                st.table(get_topk(cos_sim(vec, Q), questions, K))
            with right:
                st.header('Matchande svar')
                st.table(get_topk(sm_sim(vec, A), answers, K))
        elif which == 'Svar':
            vec = model.generate_right((ixs, att))[0]
            left, right = st.beta_columns(2)
            with left:
                st.header('Liknande svar')
                st.table(get_topk(cos_sim(vec, A), answers, K))
            with right:
                st.header('Matchande frågor')
                st.table(get_topk(sm_sim(vec, Q), questions, K))


main()

