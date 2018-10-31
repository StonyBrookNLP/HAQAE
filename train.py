########################################
#   module for training the DAVAE model
#
#
########################################
import torch 
import torch.nn as nn
from torchtext.data import Iterator as BatchIter
import argparse
import numpy as np
import random
import math
from torch.autograd import Variable
from EncDec import Encoder, Decoder, Attention, fix_enc_hidden, kl_divergence
import torch.nn.functional as F
import data_utils as du
from DAVAE import DAVAE
from DAG import example_tree
from masked_cross_entropy import masked_cross_entropy
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK
import time
from torchtext.vocab import GloVe
import pickle
import gc
import glob
import sys
import os

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)



def monolithic_compute_loss(iteration, model, target, target_lens, latent_values, latent_root, diff, dec_outputs, use_cuda, args, train=True): 
    """
    use this function for validation loss. NO backprop in this function.
    """
    # dec_outputs is [seq_len * batch_size * decoder_hidden*size]
    # logits is [seq_len * batch_size * vocab_size]

    #sum together means
    reconstruct= torch.cat([x[1].unsqueeze(dim=0) for x in diff], dim=0)
    commit = torch.cat([x[0].unsqueeze(dim=0) for x in diff], dim=0)
    commit2 = torch.cat([x[2].unsqueeze(dim=0) for x in diff[1:]], dim=0)

    # logits is [seq_len * batch_size * vocab_size]
    logits = model.logits_out(dec_outputs) 
    logits = logits.transpose(0,1).contiguous() # convert to [batch, seq, vocab]

    if use_cuda:
        ce_loss = masked_cross_entropy(logits, Variable(target.cuda()), Variable(target_lens.cuda()))
    else:
        ce_loss = masked_cross_entropy(logits, Variable(target), Variable(target_lens))
 
    loss = ce_loss + args.commit_c*commit.mean() + reconstruct.mean() + args.commit2_c*commit2.mean() 
 
    if train:
        # if training then print stats and return total loss
        print_iter_stats(iteration, loss, ce_loss, commit, commit2, reconstruct, args, model.latent_root)
    
    return loss, ce_loss # tensor 

    


def print_iter_stats(iteration, loss, ce_loss, commit, commit2, reconstruct, args, latent_root):

    if iteration % args.log_every == 0 and iteration != 0:
        print("Iteration: ", iteration) 
        print("Total: ", loss.cpu().data[0])
        print("CE: ", ce_loss.cpu().data[0])
        print("Commit: ", commit.cpu().mean().data[0])
        print("Commit2: ", commit2.cpu().mean().data[0])
        print("Reconstruct: ", reconstruct.cpu().mean().data[0]) 
        print(latent_root.argmins[:3].data.cpu().numpy())
        print(latent_root.children[0].argmins[:3].data.cpu().numpy())
        #print(reconstruct[:2, :5])



def check_save_model_path(save_model):
    save_model_path = os.path.abspath(save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


        
def classic_train(args):
    """
    Train the model in the ol' fashioned way, just like grandma used to
    Args
        args (argparse.ArgumentParser)
    """
    if args.cuda and torch.cuda.is_available():
        print("Using cuda")
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        use_cuda = False
    else:
        use_cuda=False

    #Load the data
    print("\nLoading Vocab")
    vocab = du.load_vocab(args.vocab)
    print("Vocab Loaded, Size {}".format(len(vocab.stoi.keys())))

    if args.use_pretrained:
        pretrained = GloVe(name='6B', dim=args.emb_size, unk_init=torch.Tensor.normal_)
        vocab.load_vectors(pretrained)
        print("Vectors Loaded")

    #Set add_eos to false if you want to decode arbitrarly long conditioned on the latents (done in paper), recommended to set this to false if generating
    #event sequences (since length is not that important and we dont need the latents capturing it), if generating raw text its probably better to have it on
    #In the DAVAE class there is a train() fuction that also takes in add_eos, it should match this one
    print("Loading Dataset")
    dataset = du.SentenceDataset(args.train_data, vocab, args.src_seq_length, add_eos=True) 
    print("Finished Loading Dataset {} examples".format(len(dataset)))
    batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=True, sort_within_batch=True, device=-1)
    data_len = len(dataset)

    if args.load_model:
        print("Loading the Model")
        model = torch.load(args.load_model)
    else:
        print("Creating the Model")
        bidir_mod = 2 if args.bidir else 1
        latents = example_tree(args.num_latent_values, (bidir_mod*args.enc_hid_size, args.latent_dim), use_cuda=use_cuda) #assume bidirectional
        hidsize = (args.enc_hid_size, args.dec_hid_size)
        model = DAVAE(args.emb_size, hidsize, vocab, latents, layers=args.nlayers, use_cuda=use_cuda, pretrained=args.use_pretrained, dropout=args.dropout)

    #create the optimizer
    if args.load_opt:
        print("Loading the optimizer state")
        optimizer = torch.load(args.load_opt)
    else:
        print("Creating the optimizer anew")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time() #start of epoch 1
    curr_epoch = 1
    valid_loss = [0.0]
    for iteration, bl in enumerate(batches): #this will continue on forever (shuffling every epoch) till epochs finished
        batch, batch_lens = bl.text
        target, target_lens = bl.target 

        if use_cuda:
            batch = Variable(batch.cuda())
        else:
            batch = Variable(batch)

        model.zero_grad()
        latent_values, latent_root, diff, dec_outputs = model(batch, batch_lens)
        # train set to True so returns total loss
        loss, _ = monolithic_compute_loss(iteration, model, target, target_lens, latent_values, latent_root, diff, dec_outputs, use_cuda, args=args)
 
        # backward propagation
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # Optimize
        optimizer.step() 

        # End of an epoch - run validation
        if ((args.batch_size * iteration) % data_len == 0 or iteration % args.validate_after == 0) and iteration != 0:
            print("\nFinished Training Epoch/iteration {}/{}".format(curr_epoch, iteration))

            # do validation
            print("Loading Validation Dataset.")
            val_dataset = du.SentenceDataset(args.valid_data, vocab, args.src_seq_length, add_eos=True) 
            print("Finished Loading Validation Dataset {} examples.".format(len(val_dataset)))
            val_batches = BatchIter(val_dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, sort_within_batch=True, device=-1)
            valid_loss = 0.0
            for v_iteration, bl in enumerate(val_batches):
                batch, batch_lens = bl.text
                target, target_lens = bl.target
                batch_lens = batch_lens.cpu()
                if use_cuda:
                    batch = Variable(batch.cuda(),volatile=True)
                else:
                    batch = Variable(batch, volatile=True)

                latent_values, latent_root, diff, dec_outputs = model(batch, batch_lens) 
                # train set to False so returns only CE loss
                loss, ce_loss = monolithic_compute_loss(iteration, model, target, target_lens, latent_values, latent_root, diff, dec_outputs, use_cuda, args=args, train=False)
                valid_loss = valid_loss + ce_loss.data.clone()

            valid_loss = valid_loss/(v_iteration+1)   
            print("**Validation loss {:.2f}.**\n".format(valid_loss[0]))

            # Check max epochs and break
            if (args.batch_size * iteration) % data_len == 0:
                curr_epoch += 1
            if curr_epoch > args.epochs:
                print("Max epoch {}-{} reached. Exiting.\n".format(curr_epoch, args.epochs))
                break

        # Save the checkpoint
        if iteration % args.save_after == 0 and iteration != 0: 
            print("Saving checkpoint for epoch {} at {}.\n".format(curr_epoch, args.save_model))
            # curr_epoch and validation stats appended to the model name
            torch.save(model, "{}_{}_{}_.epoch_{}.loss_{:.2f}.pt".format(args.save_model, args.commit_c, args.commit2_c,curr_epoch, float(valid_loss[0])))
            torch.save(optimizer, "{}.{}.epoch_{}.loss_{:.2f}.pt".format(args.save_model, "optimizer", curr_epoch, float(valid_loss[0])))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DAVAE')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--vocab', type=str, help='the vocabulary pickle file')
    parser.add_argument('--emb_size', type=int, default=300, help='size of word embeddings')
    parser.add_argument('--enc_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--dec_hid_size', type=int, default=512, help='size of encoder hidden')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--log_every', type=int, default=200)
    parser.add_argument('--save_after', type=int, default=500)
    parser.add_argument('--validate_after', type=int, default=2500)
    parser.add_argument('--optimizer', type=str, default='adam', help='adam, adagrad, sgd')
    parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--seed', type=int, default=11, help='random seed') 
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--bidir', type=bool, default=True, help='Use bidirectional encoder') 
    parser.add_argument('-src_seq_length', type=int, default=50, help="Maximum source sequence length")
    parser.add_argument('-max_decode_len', type=int, default=50, help='Maximum prediction length.')
    parser.add_argument('-save_model', default='model', help="""Model filename""")
    parser.add_argument('-num_latent_values', type=int, default=512, help='How many values for each categorical value')
    parser.add_argument('-latent_dim', type=int, default=256, help='The dimension of the latent embeddings')
    parser.add_argument('-use_pretrained', type=bool, default=True, help='Use pretrained glove vectors')
    parser.add_argument('-commit_c', type=float, default=0.25, help='loss hyperparameters')
    parser.add_argument('-commit2_c', type=float, default=0.15, help='loss hyperparameters')
    parser.add_argument('-dropout', type=float, default=0.0, help='loss hyperparameters')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--load_opt', type=str)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open('{}_args.pkl'.format(args.save_model), 'wb') as fi:
        pickle.dump(args, fi)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # diff between train and classic: in classic pass .txt etension for files.
    classic_train(args)



