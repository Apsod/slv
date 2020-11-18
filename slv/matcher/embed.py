import argparse
import sys
import json

import tqdm
import torch

from slv.matcher.data import KundoData, mk_loader
from slv.matcher.model import QA_model
from slv.matcher.util import mk_batch2device

def _embed(sp):
    parser = sp.add_parser('embed')
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
    parser.add_argument(
            '--out',
            default=sys.stdout,
            type=argparse.FileType('xt'))

    
    def go(args):
        ### Load the model ###
        model, tokenizer = QA_model.load(args.model_path)
        model.to(args.device)
        
        ### Load the Data ### 
        dataset = KundoData(args.data_path)

        batch_size = args.batch_size

        dl = mk_loader(
                tokenizer, dataset, 
                batch_size=args.batch_size,
                num_workers=args.data_workers,
                shuffle=False)
        

        to_device = mk_batch2device(args.device)
        

        it = tqdm.tqdm(dl)

        def decode(ixs):
            return tokenizer.decode(ixs, skip_special_tokens=True, spaces_between_special_tokens=False)
        
        ### Run the model over the data, and write the results to disk ###

        with torch.no_grad():
            for batch in map(to_device, it):
                (q, a) = batch
                qis = q[0]
                ais = a[0]

                qes = model.generate_left(q)
                aes = model.generate_right(a)

                for qe, qt, ae, at in zip(
                        map(lambda x: x.tolist(), qes), 
                        map(decode, qis), 
                        map(lambda x: x.tolist(), aes),
                        map(decode, ais)):
                    doc = {
                            'question': {
                                'embeddings': qe,
                                'text': qt,
                            },
                            'answer': {
                                'embeddings': ae,
                                'text': at,
                            }
                        }
                    args.out.write(json.dumps(doc))
                    args.out.write('\n')
    parser.set_defaults(go=go)
