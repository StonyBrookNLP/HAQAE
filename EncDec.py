#Classes for basic encoder/decoder stuff
import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncDecBase(nn.Module):
    def __init__(self, emb_size, hidden_size, embeddings=None, cell_type="GRU", layers=1, bidir=True, use_cuda=True):
            """
            For NOW, Only supporting GRU with 1 layer
            Args:
                emb_size (int) : size of input embeddings
                hidden_size (int) : size of hidden 
                embeddings (nn.Module) : Torch module (with same type signatures as nn.Embeddings) to use for embedding
                cell_type : LSTM or GRU
                bidir (bool) : Use bidirectional encoder?
            """
            super(EncDecBase, self).__init__()
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.embeddings = embeddings
            self.layers = layers
            self.bidir = bidir
            self.cell_type = cell_type
            self.use_cuda = use_cuda
            if cell_type == "LSTM":
                self.rnn = nn.LSTM(self.emb_size, self.hidden_size, self.layers, bidirectional=self.bidir, batch_first=True)
            else:
                self.rnn = nn.GRU(self.emb_size, self.hidden_size, self.layers, bidirectional=self.bidir, batch_first=True)

    def forward(input, hidden):
        raise NotImplementedError

    def initHidden(self, batch_size):
        dirs = 2 if self.bidir else 1
        if self.cell_type == "LSTM":
            hidden = (Variable(torch.zeros(batch_size, self.layers*dirs, self.hidden_size)),
                    Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size)))
        else:
            hidden = Variable(torch.zeros(self.layers*dirs, batch_size, self.hidden_size))

        if self.use_cuda:
            return hidden.cuda()
        else:
            return hidden

class Encoder(EncDecBase):
    'Encoder class to use with the DVae'
    def forward(self, input, hidden, seq_lens, use_packed=True):
        """
        Encode an entire sequence
        Args:
            input (Tensor, [batch, seq_len]) : Tensor of input ids for the embeddings lookup
            seq_lens (Tensor [seq_len]) : Store the sequence lengths for each batch for packing
            hidden (Tuple(FloatTensor)) : (h, c) if LSTM, else just h
            use_packed (bool) : whether to use a packed sequence

        Returns:
            output (Tensor, [batch, seq_len, hidden*directions) : output at each time step
            last state (Tuple(Tensor)) : (h_n, c_n) [batch, layers*directions, hidden_size]
        """
        out = self.embeddings(input).view(input.shape[0], input.shape[1], -1) #[batch, seq_len, emb_size]
        
        if use_packed:
            packed_input = pack_padded_sequence(out, seq_lens.cpu().numpy(), batch_first=True)
            self.rnn.flatten_parameters()
            packed_out, hidden = self.rnn(packed_input, hidden)
            enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            enc_out, hidden = self.rnn(out, hidden)

        return enc_out, hidden


class Decoder(EncDecBase):
    'Decoder (InputFeed) class to use with the DVae'

    def __init__(self, emb_size, hidden_size, embeddings=None, cell_type="GRU", layers=1, attn_dim=-1, use_cuda=True, dropout=0.0):

        #All the same except for attn_dim, which is the output of dim of attn, should be same as hidden_size 
        #attn_dim is tuple of (attnetion memory dim, attnention output dim)
        if attn_dim is None:
            attn_mem_dim = 2*hidden_size
            attndim = hidden_size
        else:
            attn_mem_dim, attndim = attn_dim

        bidir = False

        super(Decoder, self).__init__(emb_size + attndim, hidden_size, embeddings, cell_type, layers, bidir, use_cuda) 
        self.attn_dim = attndim
        #Previous output of attention, concat to input on text step, init to zero
        self.input_feed = None #Variable(torch.zeros(batch_size, self.attn_dim)) 
        self.attention = Attention((hidden_size, attn_mem_dim, self.attn_dim), use_cuda=self.use_cuda)
        
        if dropout > 0:
            print("Using a Dropout Value of {} in the decoder".format(dropout))
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def reset_feed_(self):
        """
        Reset the input feed so pytorch clears the memory
        """
        del self.input_feed
        self.input_feed = None

    def init_feed_(self, feed):
        """
        Initialize the input feed (usually with zero vector)
        Args
            feed (Tensor [batch_size, self.attn_dim)
        """
        if self.input_feed is None:
            self.input_feed = feed

    def forward(self, input, hidden, memory):
        """
        Run a SINGLE computation step
        Concat the inputfeed to the input,
        Update the RNN state
        Then compute the attention vector output
        Args:
            input (Tensor, [batch]) : Tensor of input ids for the embeddings lookup (one per batch since its done one at a time)
            hidden (Tuple(FloatTensor)) : (h, c) if LSTM, else just h, previous state
            memory (Tensor, [batch, num_latents, latent_dim]) : The latents to attend over (we have num_latents of them)

        Returns:
            output (Tensor, [batch, attn_dim]) : output at all time steps 
                   (the output is the output from attention (the pre logits))
            last state (Tuple(Tensor)) : (h_n, c_n) [batch, layers, hidden_size], the actual last state
        """
        if self.drop is None:
            out = self.embeddings(input).view(input.shape[0], -1) #[batch, emb_size]
        else:
            out = self.drop(self.embeddings(input).view(input.shape[0], -1)) #[batch, emb_size]

        #concat input feed 
        dec_input = torch.cat([out, self.input_feed], dim=1).unsqueeze(dim=1) #[batch, emb_size + attn_dim]
        self.rnn.flatten_parameters()
        rnn_output, hidden = self.rnn(dec_input, hidden) #rnn_output is hidden state of last layer
        #rnn_output dim is [batch, 1, hidden_size]
        rnn_output=torch.squeeze(rnn_output, dim=1)
        dec_output, scores = self.attention(rnn_output, memory)
        if self.drop is not None:
            dec_output = self.drop(dec_output)
        self.input_feed = dec_output #UPDATE Input Feed
        
       
        return dec_output, hidden

class Attention(nn.Module):
    'Bilinear Attention Form (a la Luong et al 2014)'
    def __init__(self, dim, use_cuda=True):
        """
        Args
            dim (int or Tuple) : dimension of the query vector and memory* vectors if int
                if tuple it should be a tuple (query dim, memory dim, output_dim)
                *Note I do not condone the usage of this term, it just seems to be unfortunately common, so eh, I'll use it.
        """
        super(Attention, self).__init__()
        if isinstance(dim, tuple):
            self.query_dim, self.memory_dim, self.output_dim = dim
        else:
            self.query_dim = self.memory_dim = self.output_dim = dim

        self.linear_in = nn.Linear(self.query_dim, self.memory_dim, bias=False) #this is the W for computing scores
        self.linear_out = nn.Linear(self.query_dim + self.memory_dim, self.output_dim, bias=False) #Multiply context vector concated with hidden
        self.use_cuda = use_cuda

    def forward(self, input, memory, mem_lens=None):
        """
        Computes the attention score for key e_i as (e_i^T W h), where h is the decoder hidden state,
        Takes convex combination of scores and memory states to get context vector, then concats this
        vector with the decoder state passed as input, and performs a non linear transformation, the out put of 
        this is returned
        
        Args
            input (FloatTensor, [batch, dim]) : The input (query vectors, most likely decoder hidden state)
            mem_lens (Tensor [batch]) : Store the memory lengths for each batch 
            memory (FloatTensor, [batch, seq_len, dim]) : What to attend to 
        Returns
            attn_output (FloatTensor, [batch, dim]) : The output of attn (probably use a feed input to next step)
            scores (FloatTensor, [batch, seq_len]) : The attn scores output for each batch
        """
        batch, dim = input.shape 

        Wh = self.linear_in(input).unsqueeze(1) #[batch, 1, mem_dim]
        memory_t = memory.transpose(1,2) #[batch, dim, seq_len] 
        scores = torch.bmm(Wh, memory_t) #[batch, 1, seq_len]

        if mem_lens is not None: #mask out the pads
            mask = sequence_mask(mem_lens)
            if self.use_cuda:
                mask = mask.unsqueeze(1).cuda()  # Make broadcastable.
            else:
                mask = mask.unsqueeze(1)
            scores.data.masked_fill_(1 - mask, -float('inf'))

        #Important Bug fix, that took a long time to find
        #Placing comment here to mark the achievement

        if self.use_cuda:
            scale = Variable(torch.Tensor([math.sqrt(memory.shape[2])]).view(1,1,1).cuda())
        else:
            scale = Variable(torch.Tensor([math.sqrt(memory.shape[2])]).view(1,1,1))

        scores = F.softmax(scores/scale, dim=2) #[batch, 1, seq_len], scores for each batch 

        context = torch.bmm(scores, memory).squeeze(dim=1) #[batch, dim], context vectors for each batch
        cat = torch.cat([context, input], 1) 
        attn_output = F.tanh(self.linear_out(cat))

        return attn_output, scores

def gather_last(input, lengths, use_cuda=True):
    """
    In a padded batched sequence, gather the last 
    element for each according to lengths
    Args
        input (Tensor [batch, seq_len, *]) : padded input
        dim (int) dimension to gather on
        lengths (Tensor [batch]): lengths of each batch
    Returns 
        Tensor [batch, 1, *]
    """
    #print("seq len type {}".format(type(lengths)))
    index_vect = torch.max(torch.LongTensor(lengths.shape).zero_(), lengths - 1).view(lengths.shape[0], 1,1) #convert len to index
    index_tensor = torch.LongTensor(input.shape[0], 1, input.shape[2]).zero_() + index_vect
    if use_cuda:
        return torch.gather(input, 1, Variable(index_tensor.cuda()))
    else:
        return torch.gather(input, 1, Variable(index_tensor))

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Taken From OpenNMT code
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def fix_enc_hidden(h):
    # The encoder hidden is (layers*directions) x batch x dim.
    # We need to convert it to  layers x batch x (directions*dim).
    #THIS ASSUMES h COMES FROM A BIDERECTIONAL (directions=2) ENCODER
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

def kl_divergence(q, p=None, use_cuda=True):
    """
    Compute the kl divergence, KL[q(z) || p(z)]
    between two multinomials. Assume p to be uniform if none
    Args:
        q (Variable torch.Tensor [*, dim])
        p Variable torch.Tensor([1,dim])
    """
    dim = q.shape[1]
    if p is None:
        a =torch.zeros(1,dim) + 1.0/dim
        if use_cuda:
            p = Variable(a.cuda())
        else:
            p = Variable(a)

    return torch.sum(q*(torch.log(q)-torch.log(p)), dim=1)




