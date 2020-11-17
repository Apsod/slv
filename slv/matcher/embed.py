import torch
import json
import tqdm
from slv.matcher.data import KundoData, mk_loader
from slv.matcher.model import QA_model
from slv.matcher.util import mk_batch2device

def _embed(sp):
    parser = sp.add_subparser('embed')
    parser.add_argument(
            '--model_path',
            type=str)
    parser.add_argument(
            '--data_path',
            type=str)
    parser.add_argument(
            '--batch_size',
            default=16,
            type=int)
    parser.add_argument(
            '--device',
            default=torch.device('cpu'),
            type=torch.device
            )
    parser.add_argument(
            '--data_workers',
            default=2,
            type=int
            )

    
    def go(args):
        model, tokenizer = QA_model.load(model_path)
        model.to(args.device)

        dataset = KundoData(args.data_path)

        batch_size = args.batch_size

        dl = mk_loader(
                tokenizer, dataset, 
                batch_size=args.batch_size,
                num_workers=args.data_workers,
                shuffle=False)
        

        to_device = mk_batch2device(args.device)

        it = tqdm.tqdm(dl)

        q_json = open('questions.json', 'wt')
        a_json = open('answers.json', 'wt')

        with torch.no_grad():
            for batch in map(to_device, it):
                (q, a) = batch

                qe = model.generate_left(q)

                for emb, ix in zip(qe, qi):
                    doc = {}
                    doc['embedding'] = emb.tolist()
                    doc['question'] = tokenizer.decode(ix, skip_special_tokens=True, spaces_between_special_tokens=False)
                    q_json.write(json.dumps(doc))
                    q_json.write('\n')

                ae = model.generate_right(a)

                for emb, ix in zip(ae, ai):
                    doc = {}
                    doc['embedding'] = emb.tolist()
                    doc['answer'] = tokenizer.decode(ix, skip_special_tokens=True, spaces_between_special_tokens=False)
                    a_json.write(json.dumps(doc))
                    a_json.write('\n')
    parser.set_defaults(go=go)
