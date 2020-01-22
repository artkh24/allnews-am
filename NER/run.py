"""
Usage:
    run.py train --train-src=<file>  [options]
    run.py test --test-src=<file> [options]

Options:
    -h --help                         show this screen
    --cuda                            use GPU
    --seed=<int>                      seed [default: 0]
    --train-src=<file>                train source file
    --test-src=<file>                 test source file
    --model-root=<file>               the path to the directory for model files
                                      [default: ./models]
    --batch-size=<int>                batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 1]
    --max-len=<int>                   sentence max size [default: 128]
    --lr=<float>                      learning rate [default: 3e-5]
    --train-test-split=<float>        train test split [default: 0.1]
    --full-finetuning                 use full finetuning
"""

import os
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from typing import List, Tuple, Dict, Set, Union
from seqeval.metrics import f1_score, classification_report
import numpy as np
import sys
from docopt import docopt
import sentence
from tqdm import trange
from BERT.util_ner import convert_examples_to_features, get_labels, read_examples_from_file


def loadData(tokenizer,datapath,mode,max_length,labels):
    examples = read_examples_from_file(datapath, mode=mode)
    features = convert_examples_to_features(
        examples,
        labels,
        max_length,
        tokenizer,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train(args: Dict):
    MAX_LEN = int(args['--max-len'])
    bs = int(args['--batch-size'])
    model_root = args['--model-root'] if args['--model-root'] else './models'


    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    if args['--cuda']:
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)


    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    labels = get_labels(args['--train-src'] + "/labels.txt")
    traindataset=loadData(tokenizer,args['--train-src'],'train',MAX_LEN,labels)
    devdataset = loadData(tokenizer, args['--train-src'], 'dev', MAX_LEN,labels)


    train_sampler = RandomSampler(traindataset)
    train_dataloader = DataLoader(traindataset, sampler=train_sampler, batch_size=bs)

    valid_sampler = SequentialSampler(devdataset)
    valid_dataloader = DataLoader(devdataset, sampler=valid_sampler, batch_size=bs)

    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased",num_labels=len(labels))

    if args['--cuda']:
        model.cuda()

    FULL_FINETUNING = True if args['--full-finetuning'] else False
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=float(args['--lr']))

    epochs = int(args['--max-epoch'])
    max_grad_norm = 1.0
    hist_valid_scores = []

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask,_, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask,_, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        label_map = {i: label for i, label in enumerate(labels)}
        pred_tags = [label_map[p_i] for p in predictions for p_i in p]
        valid_tags = [label_map[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        f1=f1_score(valid_tags,pred_tags)
        print("F1-Score: {}".format(f1))

        is_better = len(hist_valid_scores) == 0 or f1 > max(hist_valid_scores)
        hist_valid_scores.append(f1)
        if is_better:
            output_model_file = os.path.join(model_root, "model_file.bin")
            output_config_file = os.path.join(model_root, "config_file.bin")
            output_vocab_file = model_root

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_vocab_file)

    print('reached maximum number of epochs!', file=sys.stderr)
    exit(0)


def evaluate(args:Dict):
    model_root = args['--model-root'] if args['--model-root'] else './models'
    print("load model from {}".format(model_root), file=sys.stderr)

    labels = get_labels(args['--test-src'] + "/labels.txt")


    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    output_model_file = os.path.join(model_root, "model_file.bin")
    output_config_file = os.path.join(model_root, "config_file.bin")
    output_vocab_file = os.path.join(model_root, "vocab.txt")
    config = BertConfig.from_json_file(output_config_file)
    model = BertForTokenClassification(config,num_labels=len(labels))
    state_dict = torch.load(output_model_file)
    model.load_state_dict(state_dict)
    tokenizer = BertTokenizer(output_vocab_file, do_lower_case=False)


    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    MAX_LEN = int(args['--max-len'])
    testdataset = loadData(tokenizer, args['--test-src'], 'test', MAX_LEN, labels)

    test_sampler = SequentialSampler(testdataset)
    test_dataloader = DataLoader(testdataset, sampler=test_sampler, batch_size=int(args['--batch-size']))

    model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask,_, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    label_map = {i: label for i, label in enumerate(labels)}
    pred_tags = [[label_map[p_i] for p_i in p] for p in predictions]
    test_tags = [[label_map[l_ii] for l_ii in l_i] for l in true_labels for l_i in l]


    print("Test loss: {}".format(eval_loss / nb_eval_steps))
    print("Test Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Test F1-Score: {}".format(f1_score(test_tags, pred_tags)))


def main():
    """ Main func.
    """
    args = docopt(__doc__)
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['test']:
        evaluate(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()