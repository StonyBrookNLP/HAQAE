##################################
#   The DAG structures for the latent space
#   Currently this is just a tree, sorry for the false advertising (can be extended to a dag latter on)
##################################
import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from EncDec import Encoder, Decoder, Attention, gather_last
import torch.nn.functional as F


class LatentNode(nn.Module):
    'a node in the latent dag graph, represents a latent variable'
    def __init__(self, K, dim, nodeid="0", embeddings=None, use_attn=True, use_cuda=True, nohier_mode=False):
        """
        Args
            K (int) : number of latent categorical values this can take on (and thus, the # of embeddings)
            dim (int tuple) :  (query dimension, encoder input (memory) dimension, latent embedding dimension (output))
            nodeid (str) : an optional id for naming the node
            embeddings (nn.Embeddings) : Pass these if you want to create the embeddings, else just go with default
            nohier_mode (bool) : Run the NOHIER model instead
        """
        super(LatentNode, self).__init__()
        self.children = []  #list of LatentNodes
        self.parents = []
        #Value is a batch sized list of (Variable) torch.Tensor equal to the embedding for the category this variable currently has taken on
        self.value = None 
        self.diffs = None
        self.index_value = None #Index is the indices into the embedding of the above
        self.nohier = nohier_mode

        #print("use_cuda is {}".format(use_cuda))
        
        if use_attn:
            if use_cuda:
                self.attn = Attention(dim).cuda()
            else:
                self.attn = Attention(dim, use_cuda=False)
        else:
            self.attn=None

        self.nodeid=nodeid
        self.K = K
        self.dim = dim[2] #dimension of the embeddings for the latent nodes
        self.all_dims = dim
        
        if embeddings is not None:
            if use_cuda:
                self.embeddings = embeddings.cuda()
            else:
                self.embeddings = embeddings
        else:
            if use_cuda:
                self.embeddings = nn.Embedding(K, self.dim).cuda()
            else:
                self.embeddings = nn.Embedding(K, self.dim)

            self.embeddings.weight.data = torch.nn.init.xavier_uniform(self.embeddings.weight.data)

        #Don't forget to initialize weights

    def isroot(self):
        return self.parents == []
    def isleaf(self):
        return self.children == []

    def add_child_(self, child):
        """
        Args
            child (LatenNode) : Latent node to add as a child to this node
        """
        child.parents.append(self)
        self.children.append(child)
        self.add_module(child.nodeid, child) #This is important so the children acutally get updated

    def prune_(self):
        """
        Prune embeddings for self and children
        """
        self.embeddings.weight = nn.Parameter(prune_latents(self.embeddings.weight.data, 1))
        self.K = self.embeddings.weight.data.shape[0]
        print(self.nodeid)
        print(self.K)
        for child in self.children:
            child.prune_()

    def zero_attn_grads(self):
        """
        zero out the attn grads so they are not updated twice
        """
        self.attn.zero_grad()
        for child in self.children:
            child.zero_attn_grads()


    def set_use_cuda(self, value):
        self.use_cuda = value
        self.attn.use_cuda = value
        for child in self.children:
            child.set_use_cuda(value)

    def infer_(self, input_memory, input_lens, init_query=None):
        """
        Calculate the current value of Z for this variable (deterministic sample),
        given the values of parents and the input memory, then update this variables value
        Args:
            memory (FloatTensor, [batch, seq_len, dim]) : The input encoder states (what is attended to)
            Input lens is a [batch] size Tensor with the lengths of inputs for each batch

            init_query (an initial query for the root)
        """
        #For now assume a tree structure, with only one parent

        if not self.isroot() and not self.nohier:  #if we are a child node AND we are not running in nohier mode

            prev_latent = self.parents[0].value #propogate decoder loss back through attn and any previous attns

            V, scores = self.attn(prev_latent, input_memory, input_lens) #V has shape [batch, dim]

            V2, scores2 = self.attn(self.parents[0].embeddings.weight[self.parents[0].argmins], input_memory.detach(), input_lens) 

        else:
            V, scores = self.attn(init_query, input_memory, input_lens) 

        batch_size = V.shape[0]

        #Quantization Operation
        W = self.embeddings.weight

        L2 = lambda x,y : ((x-y)**2)
        
        #Get indices of nearest embeddings
        vals, self.argmins = L2(V.contiguous().view(batch_size, 1, self.dim), W.contiguous().view(1, self.K, self.dim)).sum(2).min(1)

        self.probs = None

#        print(self.nodeid)
#        print(self.argmins[0])


        if self.isroot() or self.nohier:

            self.value = W[self.argmins].detach() + V - V.detach() #This is the straight through estimator (Confirmed that this works)

            self.diffs = (torch.sum(L2(W[self.argmins].detach(), V), dim=1), #commitement loss
                    torch.sum(L2(W[self.argmins], V.detach()), dim=1))   #reconstruct loss
        else:
            self.value = W[self.argmins].detach() + V - V.detach()

            self.diffs = (torch.sum(L2(W[self.argmins].detach(), V), dim=1), #commitment loss
                    torch.sum(L2(W[self.argmins], V.detach()), dim=1),  #reconstruct loss
                    torch.sum(L2(W[self.argmins].detach(), V2), dim=1)) #how much to change actual parent embedding



    def infer_all_(self, input_memory, input_lens, init_query=None):
        """
        Call infer recusivly down the tree, starting from this node
        Args:
            memory (FloatTensor, [batch, seq_len, dim]) : The input encoder states (what is attended to)
        """
        #For now assume a tree structure, with only one parent

        self.infer_(input_memory, input_lens, init_query)
        for child in self.children:
            child.infer_all_(input_memory, input_lens, init_query)


    def forward(self, input_memory, input_lens, init_query):
        """
        Input lens is a [batch] size Tensor with the lengths of inputs for each batch
        """
        self.infer_all_(input_memory, input_lens, init_query)
        collected = self.collect()
        diffs = self.collect_diffs()
        self.reset_values()
        return collected, diffs

    def collect_diffs(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        diff_list = [self.diffs]
        for child in self.children:
            diff_list += child.collect_diffs()

        return diff_list


    def collect(self):
        """
        Collect all the latent variable values in the tree
        Should be called from root
        Returns
            latents (Variable, [batch, num_latents, dim])
        """
        latent_list = [self.value]
        for child in self.children:
            latent_list += child.collect()

        if self.isroot():
            return torch.stack(latent_list, dim=1)
        else:
            return latent_list

    def reset_values(self):
        """
        Reset all of the values for each node (so 
        that pytorch cleans up the Variables for the next round)
        """

        self.diffs = None
        self.value = None
        for child in self.children:
            child.reset_values()

    def set_nohier(self, value=False):
        """
        Set nohier attribute to false for this node and all children.
        This is for backwards compatibility with older versions
        """
        self.nohier = value
        for child in self.children:
            child.set_nohier(value)


def example_tree(K, all_dim, use_cuda=True, nohier_mode=False):
    """
    An example function of building trees/dags to use in DAVAE
    all_dim : tuple (encoder dim, latent_dim)
    """

                #Query dim  #Mem Dim   #Latent Dim
    root_dim = (all_dim[0], all_dim[0], all_dim[1])
    if nohier_mode:
        dim = root_dim
    else:
        dim = (all_dim[1], all_dim[0], all_dim[1])

    root = LatentNode(K, root_dim, nodeid="ROOT", use_cuda=use_cuda, nohier_mode=nohier_mode)
    child_k=K

    if nohier_mode:
        print("Using NOHIER")

    #THIS WORKS FINE (Use Xavier_normal)
    print("Using Linear Chain")
    i=1
    id_str = "Level_{}".format(i)
    child1= LatentNode(child_k, dim, nodeid=id_str, use_cuda=use_cuda, nohier_mode=nohier_mode)

    i+=1
    id_str = "Level_{}".format(i)
    child2= LatentNode(child_k, dim, nodeid=id_str, use_cuda=use_cuda, nohier_mode=nohier_mode)

    i+=1
    id_str = "Level_{}".format(i)
    child3= LatentNode(child_k, dim, nodeid=id_str, use_cuda=use_cuda, nohier_mode=nohier_mode)

    i+=1
    id_str = "Level_{}".format(i)
    child4= LatentNode(child_k, dim, nodeid=id_str, use_cuda=use_cuda, nohier_mode=nohier_mode)

    child3.add_child_(child4)
    child2.add_child_(child3)
    child1.add_child_(child2)
    root.add_child_(child1)

#    for i in range(2):
#        id_str = "Level1_{}".format(i)
#        child = LatentNode(child_k, dim, nodeid=id_str, use_cuda=use_cuda)
#        child.add_child_(LatentNode(child_k, dim, nodeid=id_str + "_2", use_cuda=use_cuda))
#        root.add_child_(child)
#
    return root

def prune_latents(embeddings, threshold_norm):
    """
    Remove any embeddings with norm less than threshold_norm (these were probably not updated in training)
    args
        embeddings (Tensor [K x dim])
        threshold_norm (int)
    """
    K = embeddings.shape[0]
    indices = torch.masked_select(torch.arange(K), torch.ge(torch.norm(embeddings, 2, dim=1), threshold_norm))
    return embeddings[indices.type(torch.LongTensor)]


