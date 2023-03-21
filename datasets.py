#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''


import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import random


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression =False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data

class MaskedLMDataset(Dataset):
    def __init__(self, dataset, args, single):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mask_token = 103
        self.ignore_label = -100
        self.single = single

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def masked_data(self, data):
        if self.single:
            sents = [x[0] for x in data]
            labels = [x[1] for x in data]
            sent_ids = [x[2] for x in data]
        else:
            sents = [x[0] for x in data]
            sents2 = [x[1] for x in data]
            sents.extend(sents2)
            sent_ids = [x[2] for x in data]

        # Compute the number of tokens to mask
        num_tokens = sum([len(self.tokenizer.encode(sent)) for sent in sents])
        num_tokens_to_mask = int(num_tokens * 0.15)

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = token_ids.detach().clone()

        # create a set of forbidden values
        forbidden_values = {101, 102, 0}

        # randomly select indices to mask
        random_indices = []
        for i in range(len(sents)):
            indices = torch.arange(len(token_ids[i]))
            mask = torch.ones_like(token_ids[i], dtype=torch.bool)
            for v in forbidden_values:
                mask &= (token_ids[i] != v)
            indices = indices[mask]
            num_tokens_i = len(indices)
            num_tokens_to_mask_i = int(num_tokens_to_mask * num_tokens_i // num_tokens)
            if num_tokens_to_mask_i > 0:
                selected_indices = torch.randperm(num_tokens_i)[:num_tokens_to_mask_i]
                random_indices.append(indices[selected_indices])
            else:
                random_indices.append(torch.tensor([], dtype=torch.long))

        # create a mask with all False values except for the randomly selected indices
        BERT_mask = torch.zeros(len(sents), len(token_ids[0]), dtype=torch.bool)
        for i, indices in enumerate(random_indices):
            BERT_mask[i, indices] = True
        


        selection = []
        for i in range(token_ids.shape[0]):
            selection.append(torch.flatten(BERT_mask[i].nonzero()).tolist())

        for i in range(token_ids.shape[0]):
            token_ids[i, selection[i]] = self.mask_token
        for i in range(labels.shape[0]):
            labels[i, selection[i]] = self.ignore_label

    

        return token_ids, attention_mask, labels, sents, sent_ids, BERT_mask

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids, BERT_mask = self.masked_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids,
                'bert_mask': BERT_mask
            }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data



def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data
