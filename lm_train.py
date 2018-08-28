# coding: utf-8

import argparse
import time
import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from masked_cross_entropy import masked_cross_entropy
from torchtext.data import Iterator as BatchIter
from LSTMLM import LSTMLM
import data_utils as du
from torchtext.vocab import GloVe
import sys

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)


def do_training(use_cuda=True):

    # Using our data utils to load data
    vocab = du.load_vocab(args.vocab)
    nvocab = len(vocab.stoi.keys())
    print("*Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))
 

    if args.pretrained:
        print("using pretrained vectors.") 
        pretrained = GloVe(name='6B', dim=args.emsize, unk_init=torch.Tensor.normal_)
        vocab.load_vectors(pretrained)
        print("Vectors Loaded")
 
    if args.emb_type:
        vocab2 = du.load_vocab(args.vocab2)
        nvocab2 = len(vocab2.stoi.keys())
        print("*Vocab2 Loaded, Size {}".format(len(vocab2.stoi.keys())))

        dataset = du.LMRoleSentenceDataset(args.train_data, vocab, args.train_type_data, vocab2) 
        print("*Train Dataset Loaded {} examples".format(len(dataset))) 

        # Build the model: word emb + type emb  
        model = LSTMLM(args.emsize, args.nhidden, args.nlayers, nvocab, pretrained=args.pretrained, vocab=vocab, type_emb=args.emb_type, ninput2=args.em2size, nvocab2=nvocab2, dropout=args.dropout, use_cuda=use_cuda)
        print("Building word+type emb model.")

    else:

        dataset = du.LMSentenceDataset(args.train_data, vocab) 
        print("*Train Dataset Loaded {} examples".format(len(dataset))) 

        # Build the model: word emb
        model = LSTMLM(args.emsize, args.nhidden, args.nlayers, nvocab, pretrained=args.pretrained, vocab=vocab, dropout=args.dropout, use_cuda=use_cuda)
        print("Building word emb model.")


    data_len = len(dataset)
    batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=True, sort_within_batch=True, device=-1) 

    ## some checks
    tally_parameters(model)

    if use_cuda:
        model=model.cuda()

    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_loss = [0.0]

    # DO TRAINING   
    
    total_loss = 0.0
    lapse = 1     
    faulty = False
    for iteration, bl in enumerate(batches):
        
        # batch is [batch_size, seq_len]
        batch, batch_lens = bl.text
        if args.emb_type:
            role, role_lens = bl.role

        target, target_lens = bl.target
        
        # init the hidden state before every batch
        hidden = model.init_hidden(batch.size(0)) #args.batch_size)

        # batch has SOS prepended to it.
        # target has EOS appended to it.
        if use_cuda:
            batch = Variable(batch.cuda())
            target = Variable(target.cuda())
            if args.emb_type:
                role = Variable(role.cuda())
        else:
            batch = Variable(batch) 
            target = Variable(target)
            if args.emb_type:
                role = Variable(role)

        # Repackaging is not needed.

        # zero the gradients
        model.zero_grad()
        # run the model
        logits = []
        for i in range(batch.size(1)): 
            inp = batch[:, i] 
            inp = inp.unsqueeze(1) 
            if args.emb_type:
                # handle OOI exception by breaking out of the inner loop and moving to the next.  
                try:
                    typ = role[:, i]
                    typ = typ.unsqueeze(1)
                    logit, hidden = model(inp, hidden, typ) 
                except Exception as e:
                    print("ALERT!! word and type batch error. {}".format(e)) 
                    faulty = True
                    break
            else:
                # keep updating the hidden state accordingly
                logit, hidden = model(inp, hidden)

            logits += [logit]

        # if this batch was faulty; continue to the next iteration
        if faulty:
            faulty = False 
            continue 

        # logits is [batch_size, seq_len, vocab_size]
        logits = torch.stack(logits, dim=1) 
        if use_cuda:
            loss = masked_cross_entropy(logits, target, Variable(target_lens.cuda()))
        else:
            loss = masked_cross_entropy(logits, target, Variable(target_lens))

        loss.backward()
 
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        # optimize
        optimizer.step()

        # aggregate the stats
        total_loss = total_loss + loss.data.clone()
        lapse += 1

        # print based on log interval
        if (iteration+1) % args.log_interval == 0:
            print("| iteration {} | loss {:5.2f}".format(iteration+1, loss.data[0]))

        # forcing buffers to write
        sys.stdout.flush()
		
        # saving only after specified iterations
        if (iteration+1) % args.save_after == 0: 
            # summarize every save after num iterations losses
            avg_loss = total_loss / lapse 
            print("||| iteration {} | average loss {:5.2f}".format(iteration+1, avg_loss.cpu().numpy()[0]))
            # reset values
            total_loss = 0.0
            lapse = 1

            #torch.save(model, "{}_.epoch_{}.iteration_{}.loss_{:.2f}.pt".format(args.save, curr_epoch, iteration+1, val_loss[0]))
            torch.save(model, "{}_.iteration_{}.pt".format(args.save, iteration+1)) 
            torch.save(optimizer, "{}.{}.iteration_{}.pt".format(args.save, "optimizer", iteration+1))
            print("model and optimizer saved for iteration {}".format(iteration+1))

 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,
                        help='location of the train data corpus')  
    parser.add_argument('--valid_data', type=str,
                        help='location of the valid data corpus') 
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhidden', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate') 
    parser.add_argument('--clip', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size') 
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='dropout applied to layers (0 = no dropout)')  
    parser.add_argument('--seed', type=int, default=11,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model_lm.pt',
                        help='path to save the final model')   
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file')
    parser.add_argument('--save_after', type=int, default=25000)
    parser.add_argument('--validate_after', type=int, default=10000)
    parser.add_argument('--emb_type', type=int, default=0)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--train_type_data', type=str,
                        help='location of the train type data corpus')
    parser.add_argument('--em2size', type=int, default=300,
                        help='size of type embeddings')
    parser.add_argument('--vocab2', type=str, help='the vocabulary pickle file')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.cuda and torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        use_cuda = False
    else:
        use_cuda=False

    print("Use CUDA {}".format(use_cuda))
    do_training(use_cuda)
