import math
import sys

import torch
import transformers
import tqdm

from slv.matcher.model import QA_model
from slv.matcher.data import KundoData, mk_loader
from slv.matcher.util import ExpMean, WelfordMean, mk_batch2device


def _train(sp):
    parser = sp.add_parser('train')
    parser.add_argument(
            '--model_path',
            default='KB/bert-base-swedish-cased',
            type=str)
    parser.add_argument(
            '--train_path',
            required=True,
            type=str)
    parser.add_argument(
            '--epochs',
            default=5
            , type=int)
    parser.add_argument(
            '--batch_size',
            default=16,
            type=int)
    parser.add_argument(
            '--lr',
            default=1e-5,
            type=float)
    parser.add_argument(
            '--weight_decay',
            default=1e-2,
            type=float)
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
        train(
                args.model_path,
                args.train_path,
                args.epochs,
                args.batch_size,
                args.lr,
                args.weight_decay,
                args.device,
                args.data_workers)

    parser.set_defaults(go=go)

def train(model_path, train_path, epochs, batch_size, lr, weight_decay, device, data_workers):

    ##### INITIALIZE MODEL FROM HUGGINGFACE #####

    model, tokenizer = QA_model.init(model_path)
    model = model.to(device=device)

    ##### SET UP TRAINING & VALIDATION DATA #####

    dataset = KundoData(train_path)
    train_size = math.floor(len(dataset) * 0.8)
    val_size = (len(dataset) - train_size)

    train_data, val_data = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(8392))

    train_dl = mk_loader(
            tokenizer, train_data,
            batch_size=batch_size,
            num_workers=data_workers)
    val_dl = mk_loader(
            tokenizer, val_data,
            batch_size=batch_size,
            num_workers=data_workers,
            shuffle=False)

    ##### SET UP OPTIMIZER AND LEARNING RATE SCHEDULER #####

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    TOT_STEPS = train_size * epochs
    WARMUP_STEPS = train_size // 2

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TOT_STEPS)

    ##### TRAIN MODEL #####

    to_device = mk_batch2device(device)

    # reasonable first guess at loss: ln(0.5)
    train_loss = ExpMean(math.log(0.5))
    best_loss = float('inf')

    for epoch in range(epochs):

        ## TRAIN ##
        it = tqdm.tqdm(train_dl)
        for batch in map(to_device, it):
            (q, a) = batch

            optimizer.zero_grad()
            loss = model.loss(q, a, loss='bce')
            loss.backward()
            optimizer.step()
            scheduler.step()

            ## UPDATE EXPONENTIAL MEAN LOSS ##
            with torch.no_grad():
                train_loss += loss.item()
                it.set_description('loss: {:.1e}'.format(train_loss.mean))
        model.save('latest.pt')

        ## VALIDATION ##
        it = tqdm.tqdm(val_dl)
        val_loss = WelfordMean()
        with torch.no_grad():
            losses = []
            for batch in map(to_device, it):
                (q, a) = batch
                val_loss += WelfordMean(model.loss(q, a, loss='bce').item(), len(batch))
            print('validation loss: {}'.format(val_loss.mean), file=sys.stderr)
            if loss < best_loss:
                best_loss = loss
                model.save('best.pt')
