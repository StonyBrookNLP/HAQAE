import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMLM(nn.Module): 

    def __init__(self, ninput, nhidden, nlayers, nvocab, pretrained=False, vocab=None, type_emb=False, ninput2=0, nvocab2=0, rnn_type="GRU", dropout=0.2, use_cuda=True):

        super(LSTMLM, self).__init__()

        self.dropout = nn.Dropout(dropout)
        # word embedding layer
        self.embedding = nn.Embedding(nvocab, ninput)

        if type_emb:
            #print("added another embedding")
            assert ninput2 > 0 and nvocab2 > 0, "set the emb and vocab size for word type."
            # type embedding layer
            self.embedding2 = nn.Embedding(nvocab2, ninput2)
            # sum up the embedding sizes
            ninput = ninput + ninput2
            print("RNN size {}".format(ninput))

        # check if GRU is sufficient
        self.rnn = nn.GRU(ninput, nhidden, nlayers, dropout=dropout, batch_first=True)
        # logit layer
        self.linear_out = nn.Linear(nhidden, nvocab)
        
        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.use_cuda = use_cuda
        self.type_emb = type_emb
        self.pretrained = pretrained
        self.vocab = vocab

        # init the embeddings and linear out params 
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        
        if self.pretrained:
            assert self.vocab is not None, "pass vocab obj when pretrained is True."
            # init with pretrained embeddings
            self.embedding.weight.data = self.vocab.vectors
        else:
            # pichotta: init weights with N(0, 0.1) and biases with 0
            # we do uniform
            self.embedding.weight.data.uniform_(-initrange, initrange)

        self.linear_out.bias.data.zero_()
        self.linear_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, input2=None): 
        """
        dropout is on the input and output
        """
        
        #print("forward received input {}".format(input.size()))
        #print("forward received input2 {}".format(input2.size()))

        batch_size = input.size(0)
        # word embedding
        emb = self.dropout(self.embedding(input))
        if self.type_emb:
            assert input2 is not None, "Type input cannot be None."
            # type embedding
            emb2 = self.dropout(self.embedding2(input2))
            #final embedding : concatentae alng emb dim 
            final_emb = torch.cat((emb, emb2), 2)
            #print("updated final emb {}".format(final_emb.size()))
            output, hidden = self.rnn(final_emb, hidden)
        else:
            #print("not updated emb {}".format(emb.size()))
            output, hidden = self.rnn(emb, hidden)

        #print("rnn output {}".format(output.size()))
        # output [batch_size, 1, hidden_size]
        output = self.dropout(output)
        # logit is [batch_size * layers, vocab]
        logit = self.linear_out(output.view(output.size(0)*output.size(1), output.size(2))) 
        return logit, hidden

    def init_hidden(self, batch_size):
        #weight = next(self.parameters())  
        hidden = Variable(torch.zeros(self.nlayers, batch_size, self.nhidden))
        if self.use_cuda:
            return hidden.cuda()
        else:
            return hidden

        return hidden
