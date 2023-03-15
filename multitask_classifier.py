import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data, MaskedLMDataset

from evaluation import model_eval_sst, test_model_multitask


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
N_SIMILARITY_CLASSES = 6
N_TASKS = 3


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        self.num_labels = 5
        self.sentiments_proj = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Linear layer that projects two concatenated [CLS] embeddings into dimensions of just one [CLS] embedding
        self.para_proj = nn.Linear(config.hidden_size * 2, 1)
        self.sim_proj = nn.Linear(config.hidden_size * 2, 1)
        self.cos = nn.CosineSimilarity()
        self.relu = nn.ReLU()
        # Linear layer for Classification Objective Function (SBERT Paper)
        self.linear_Wt = nn.Linear(3 * config.hidden_size, 5)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        first_tk = self.bert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        return first_tk


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        first_tk = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        first_tk = self.dropout(first_tk)
        first_tk = self.sentiments_proj(first_tk)
        return first_tk


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        first_tk_1 = self.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
        first_tk_2 = self.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
        output = torch.cat((first_tk_1, first_tk_2), 1)
        output = self.dropout(output)
        output = self.para_proj(output)
        return output
        


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        first_tk_1 = self.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
        first_tk_2 = self.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
        output = torch.cat((first_tk_1, first_tk_2), 1)
        # output = self.dropout(output)
        # output = self.sim_proj(output)
        output = self.cos(first_tk_1, first_tk_2)
        output = self.relu(output)
        return output


class PretrainedDataBERT(nn.Module):
    '''
        For pretraining BERT on additional datasets
    '''
    def __init__(self, config):
        super(PretrainedDataBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def predict_masked_tokens(self, input_ids, attention_mask): 
        # Basic MLM architecture
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.linear(hidden_states)
        prediction_scores = torch.nn.functional.log_softmax(hidden_states, dim=-1)
        return prediction_scores




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def pretrain_task(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')

    # SST
    sst_train_data = MaskedLMDataset(sst_train_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'vocab_size': 30522,
              'option': args.option}

    config = SimpleNamespace(**config)

    model = PretrainedDataBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_train_acc = 0

    pretrain_file_path = "~/Github/CS224n-Default-Final-Project"
    loss_fn = nn.NLLLoss(ignore_index=-100)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_masked_tokens(b_ids, b_mask)
            loss = loss_fn(logits.view(-1, config.vocab_size), b_labels.view(-1)) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)

        # if train_acc > best_train_acc:
        #     best_train_acc = train_acc
        #     save_model(model, optimizer, args, config, pretrain_file_path)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # SST
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # Paraphase
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    # STS
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for sst_train, para_train, sts_train in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader),
                                                     total=min([len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)]),
                                                     desc=f'train-{epoch}', disable=TQDM_DISABLE):
            iter_loss = 0
            
            # zero out gradients
            optimizer.zero_grad()
            
            # SST
            sst_b_ids, sst_b_mask, sst_b_labels = (sst_train['token_ids'],
                                       sst_train['attention_mask'], sst_train['labels'])

            sst_b_ids = sst_b_ids.to(device)
            sst_b_mask = sst_b_mask.to(device)
            sst_b_labels = sst_b_labels.to(device)

            logits = model.predict_sentiment(sst_b_ids, sst_b_mask)
            loss = F.cross_entropy(logits, sst_b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()

            iter_loss += loss.item()
            num_batches += 1

            # Paraphrase
            para_b_ids_1, para_b_ids_2, para_b_mask_1, para_b_mask_2, para_b_labels = (para_train['token_ids_1'], para_train['token_ids_2'], 
                                                              para_train['attention_mask_1'], para_train['attention_mask_2'],
                                                              para_train['labels'])

            para_b_ids_1 = para_b_ids_1.to(device)
            para_b_ids_2 = para_b_ids_2.to(device)
            para_b_mask_1 = para_b_mask_1.to(device)
            para_b_mask_2 = para_b_mask_2.to(device)
            para_b_labels = para_b_labels.to(device)

            logits = model.predict_paraphrase(para_b_ids_1, para_b_mask_1, para_b_ids_2, para_b_mask_2)
            # check where to put sigmoid (if anywhere)
            normalized_logits = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(torch.squeeze(normalized_logits, dim=1), para_b_labels.view(-1).float(), reduction='sum') / args.batch_size

            loss.backward()

            iter_loss += loss.item()
            num_batches += 1

            # STS
            sts_b_ids_1, sts_b_ids_2, sts_b_mask_1, sts_b_mask_2, sts_b_labels = (sts_train['token_ids_1'], sts_train['token_ids_2'], 
                                                              sts_train['attention_mask_1'], sts_train['attention_mask_2'],
                                                              sts_train['labels'])

            sts_b_ids_1 = sts_b_ids_1.to(device)
            sts_b_ids_2 = sts_b_ids_2.to(device)
            sts_b_mask_1 = sts_b_mask_1.to(device)
            sts_b_mask_2 = sts_b_mask_2.to(device)
            sts_b_labels = sts_b_labels.to(device)

            
            first_tk_1 = model.forward(input_ids=sts_b_ids_1, attention_mask=sts_b_mask_1)
            first_tk_2 = model.forward(input_ids=sts_b_ids_2, attention_mask=sts_b_mask_2)
            # sts_b_labels = (sts_b_labels>3).float()
            # sts_b_labels[sts_b_labels == 0] = -1
            
            loss = F.cosine_embedding_loss(first_tk_1, first_tk_2, sts_b_labels.view(-1)) / args.batch_size
            loss.requires_grad = True

            loss.backward()

            iter_loss += loss.item()
            num_batches += 1
            

            optimizer.step()
            train_loss += iter_loss / N_TASKS
     

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    pretrain_task(args)
    # train_multitask(args)
    test_model(args)
