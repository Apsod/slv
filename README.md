# SLV Question Answering

This repo contains helpers for extracting data from Kundo, and training Transformer-based Question Answering-Retrieval models on this data.

## Requirements

- transformers
- tqdm
- pytorch
- streamlit (optional, for demo)

## Usage

### Installation 

In this directory, install the package using pip: 

```bash
pip install . 
```

This will install the package, and create an entry point `slv`.

### Getting data from Kundo

To extract data from Kundo, use the `slv get` command: 

```bash
slv get all --out /path/to/output/file --kundo_key KUNDO_API_KEY
```

or, if you have the API key in the environment variable KUNDO: 

```bash
slv get all --out /path/to/output/file
```

For evaluation purposes, there is also a command `slv get split`, which separates the data in two: a training set, and a test set. The test set consists of all documents tagged with special tags (those supplied by Ingela, found in `slv.TAG_SET`), and the training set consists of all other documents.

(See: [](slv/kundo.py) and [](slv/__main__.py))

### Training a Model

To train a Question-Answering Retrieval model, use the `slv model train` command.

```bash
slv model train --model_path KB/bert-base-swedish-cased --train_path /path/to/slv.train.json --epochs 5 --device cuda
```

The above will train a QA-model --- initialized using KB-bert --- on the Question-Answer data in /path/to/slv.train.json, for 5 epochs, on the GPU. During training it will write two models to disk: `latest.pt`, and `best.pt`. `latest.pt` is the latest model, whereas `best.pt` is the model that performs best on the validation data (80-20 train-validation split).


```bash
usage: slv model train [-h] [--model_path MODEL_PATH] --train_path TRAIN_PATH              
                       [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                       [--weight_decay WEIGHT_DECAY] [--device DEVICE]
                       [--data_workers DATA_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --train_path TRAIN_PATH
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
  --device DEVICE
  --data_workers DATA_WORKERS
```

`model_path` is the pretrained huggingface model to load (e.g. `KB/bert-base-swedish-cased`)

`train_path` is the path to the trainig data file (extracted using `slv get`)

`device` is the device to train on (`cuda` or `cpu`)

`data_workers` is the number of workers to use for data loading (uses sensible defaults)

`--epochs`, `--batch_size`, `--lr`, `--weight_decay` are training hyperparameters: `epochs` are the number of passes over the training data we train for, `batch_size` is the size of each batch (limited by GPU memory), `lr` is the learning rate used by the optimizer, and `weight_decay` is the weight decay used by the optimizer. All these have sensible defaults, but you can experiment with them. 

(The model definition can be found in [](slv/matcher/model.py), and the training code in [](slv/matcher/trainer.py))

### Embedding questions and answers

To use a trained model to embed questions and answers, use the `slv model embed` command. 

```bash
slv model embed --model_path best.pt --data_path slv.all.json --device cuda --out embeddings.json
```

The above will load the trained model `best.pt`, and create question/answer-representations of the questions and answers in `slv.all.json`, and output these to the file `embeddings.json`.
The resulting embeddings are such that the dot product between a question and an answer should be high if the answer matches the question, and low otherwise. The result is that the 
cosine similarity between questions is a reasonable measure of similar questions, and vice versa for answers. 

```bash
usage: slv model embed [-h] [--model_path MODEL_PATH] [--data_path DATA_PATH]
                       [--batch_size BATCH_SIZE] [--device DEVICE]
                       [--data_workers DATA_WORKERS] [--out OUT]

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
  --data_path DATA_PATH
  --batch_size BATCH_SIZE
  --device DEVICE
  --data_workers DATA_WORKERS
  --out OUT
```

(the embedding code can be found in [](slv/matcher/embed.py))

### Running the streamlit demo

Given that you have a trained model, and embeddings, you can run the streamlit demo. The demo makes it possible to search for similar questions, and matching answers, to a given question (or vice versa: similar answers, and matching questions).

```bash
streamlit run path/to/slv/matcher/streamlit_app.py --model_path path/to/model.pt --embeddings path/to/embeddings.json
```
