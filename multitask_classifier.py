import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data, MaskedLMDataset

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask, model_eval_pretrain

sys.path.append('./pcgrad')
from pcgrad import PCGrad


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
pretrain_file_path="/home/ubuntu/Github/CS224n-Default-Final-Project/MLMModel.pt"


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config, pretrain_file_path):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        with open(pretrain_file_path, 'rb') as f:
            self.bert = pickle.load(f)
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
        hidden_states_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)['last_hidden_state']
        hidden_states_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)['last_hidden_state']
        mean_pooled_output_1 = torch.mean(hidden_states_1, dim=1)
        mean_pooled_output_2 = torch.mean(hidden_states_2, dim=1)
        output = self.cos(mean_pooled_output_1, mean_pooled_output_2)
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
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def predict_masked_tokens(self, input_ids, attention_mask): 
        # Basic MLM architecture
        hidden_states = self.forward(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear(hidden_states)
        prediction_scores = torch.nn.functional.log_softmax(hidden_states, dim=-1)
        return prediction_scores

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output




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
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # SST
    sst_train_data = MaskedLMDataset(sst_train_data, args, True)
    sst_dev_data = MaskedLMDataset(sst_dev_data, args, True)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)

    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_dev_data.collate_fn)
    

    # Para
    para_train_data = MaskedLMDataset(para_train_data, args, False)
    para_dev_data = MaskedLMDataset(para_dev_data, args, False)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_dev_data.collate_fn)
    
    # STS
    sts_train_data = MaskedLMDataset(sts_train_data, args, False)
    sts_dev_data = MaskedLMDataset(sts_dev_data, args, False)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)


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
    best_dev_acc = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # Think about CrossEntropy

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for sst_train, para_train, sts_train in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader),
                                                     total=min([len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)]),
                                                     desc=f'train-{epoch}', disable=TQDM_DISABLE):
            
            #SST
            sst_b_ids, sst_b_mask, sst_b_labels = (sst_train['token_ids'],
                                       sst_train['attention_mask'], sst_train['labels'])

            sst_b_ids = sst_b_ids.to(device)
            sst_b_mask = sst_b_mask.to(device)
            sst_b_labels = sst_b_labels.to(device)

            optimizer.zero_grad()
            logits = model.predict_masked_tokens(sst_b_ids, sst_b_mask)
            loss = loss_fn(logits.view(-1, config.vocab_size), sst_b_labels.view(-1)) / args.batch_size

            loss.backward()
            num_batches += 1
            train_loss += loss.item()
            
            #Para
            para_b_ids, para_b_mask, para_b_labels = (para_train['token_ids'], 
                                                      para_train['attention_mask'], para_train['labels'])

            para_b_ids = para_b_ids.to(device)
            para_b_mask = para_b_mask.to(device)
            para_b_labels = para_b_labels.to(device)

            logits = model.predict_masked_tokens(para_b_ids, para_b_mask)
            loss = loss_fn(logits.view(-1, config.vocab_size), para_b_labels.view(-1)) / args.batch_size

            loss.backward()
            num_batches += 1
            train_loss += loss.item()

            # STS
            sts_b_ids, sts_b_mask, sts_b_labels = (sts_train['token_ids'], sts_train['attention_mask'], sts_train['labels'])

            sts_b_ids = sts_b_ids.to(device)
            sts_b_mask = sts_b_mask.to(device)
            sts_b_labels = sts_b_labels.to(device)

            logits = model.predict_masked_tokens(sts_b_ids, sts_b_mask)
            loss = loss_fn(logits.view(-1, config.vocab_size), sts_b_labels.view(-1)) / args.batch_size

            loss.backward()
            num_batches += 1
            train_loss += loss.item()

            optimizer.step()

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_ = model_eval_pretrain(sst_train_dataloader, model, device)
        dev_acc, train_f1, *_ = model_eval_pretrain(sst_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            with open(pretrain_file_path, 'wb') as f:
                pickle.dump(model, f)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}")

## Currently only trains on sst dataset
def train_multitask(args, pretrain_file_path):
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

    model = MultitaskBERT(config, pretrain_file_path)
    model = model.to(device)

    lr = args.lr
    optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    best_dev_score = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        sst_y_true = []
        sst_y_pred = []
        para_y_true = []
        para_y_pred = []
        sts_y_true = []
        sts_y_pred = []

        for sst_train, para_train, sts_train in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader),
                                                     total=min([len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader)]),
                                                     desc=f'train-{epoch}', disable=TQDM_DISABLE):
            iter_loss = 0
            losses = []
            
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

            losses.append(loss)
            iter_loss += loss.item()
            num_batches += 1

            # update predicted/actual lists
            y_hat = logits.detach().argmax(dim=-1).flatten().cpu().numpy()
            b_labels = sst_b_labels.detach().flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)

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
            normalized_logits = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(torch.squeeze(normalized_logits, dim=1), para_b_labels.view(-1).float(), reduction='sum') / args.batch_size

            losses.append(loss)
            iter_loss += loss.item()
            num_batches += 1

            # update predicted/actual lists
            y_hat = logits.detach().sigmoid().round().flatten().cpu().numpy()
            b_labels = para_b_labels.detach().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)

            # STS
            sts_b_ids_1, sts_b_ids_2, sts_b_mask_1, sts_b_mask_2, sts_b_labels = (sts_train['token_ids_1'], sts_train['token_ids_2'], 
                                                              sts_train['attention_mask_1'], sts_train['attention_mask_2'],
                                                              sts_train['labels'])

            sts_b_ids_1 = sts_b_ids_1.to(device)
            sts_b_ids_2 = sts_b_ids_2.to(device)
            sts_b_mask_1 = sts_b_mask_1.to(device)
            sts_b_mask_2 = sts_b_mask_2.to(device)
            sts_b_labels = sts_b_labels.to(device)

            
            logits = model.predict_similarity(sts_b_ids_1, sts_b_mask_1, sts_b_ids_2, sts_b_mask_2)
            rescaled_logits = logits * (N_SIMILARITY_CLASSES - 1)
            loss = F.mse_loss(rescaled_logits, sts_b_labels.view(-1).float(), reduction='sum') / args.batch_size

            losses.append(loss)
            iter_loss += loss.item()
            num_batches += 1

            # update predicted/actual lists
            y_hat = logits.detach().flatten().cpu().numpy()
            b_labels = sts_b_labels.detach().flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)

            # run gradient surgery backprop and update optimizer
            optimizer.pc_backward(losses)
            train_loss += iter_loss / N_TASKS
     

        train_loss = train_loss / (num_batches)

        # evaluate on training set
        train_acc_para = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        train_acc_sst = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        train_acc_sts = np.corrcoef(sts_y_pred, sts_y_true)[1][0]

        # evaluate on dev set
        dev_eval = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        dev_acc_para = dev_eval[0]
        dev_acc_sst = dev_eval[3]
        dev_acc_sts = dev_eval[6]

        # score is sum of accuracies
        train_score = train_acc_para + train_acc_sst + train_acc_sts
        dev_score = dev_acc_para + dev_acc_sst + dev_acc_sts

        # if score is best so far, save model
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            save_model(model, optimizer.optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train score :: {train_score :.3f}, dev score :: {dev_score :.3f}\ntrain acc (para) :: {train_acc_para :.3f}, train acc (sst) :: {train_acc_sst :.3f}, train acc (sts) :: {train_acc_sts :.3f}")



def test_model(args, pretrain_file_path):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config, pretrain_file_path)
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
    train_multitask(args, pretrain_file_path)
    test_model(args, pretrain_file_path)
