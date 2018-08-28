################################
# Data Utils for generating
# plain old text (The Book Corpus)
# The input is assumed to be a pretokenized text file
# with a single sentence per line
#
# Uses texttorch stuff, so make sure thats installed 
################################
import torch 
import torch.nn as nn
import numpy as np
import math
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchtext.data as ttdata
import torchtext.datasets as ttdatasets
from torchtext.vocab import Vocab
from collections import defaultdict, Counter

#Reserved Special Tokens
PAD_TOK = "<pad>"
SOS_TOK = "<sos>" #start of sentence
EOS_TOK = "<eos>" #end of sentence 
UNK_TOK = "<unk>"
TUP_TOK = "<TUP>"
DIST_TOK = "<DIST>" # distractor token for NC task

#These are the values that should be used during evalution to keep things consistent
MIN_EVAL_SEQ_LEN = 8
MAX_EVAL_SEQ_LEN = 50 

#A Field for a single sentence from the Book Corpus (or any other corpus with a single item per line)

#PAD has an id of 1
#UNK has id of 0

def create_vocab(filename, max_size=None, min_freq=1, savefile=None, specials = [UNK_TOK, PAD_TOK, SOS_TOK, EOS_TOK]):
    """
    Create a vocabulary object
    Args
        filename (str) : filename to induce vocab from
        max_size (int) : max size of the vocabular (None = Unbounded)
        min_freq (int) : the minimum times a word must appear to be 
        placed in the vocab
        savefile (str or None) : file to save vocab to (return it if None)
        specials (list) : list of special tokens 
    returns Vocab object
    """
    count = Counter()
    with open(filename, 'r') as fi:
        for line in fi:
            for tok in line.split(" "):
                count.update([tok.rstrip('\n')])

    voc = Vocab(count, max_size=max_size, min_freq=min_freq, specials=specials)
    if savefile is not None:
        with open(savefile, 'wb') as fi:
            pickle.dump(voc, fi)
        return None
    else:
        return voc


def load_vocab(filename):
    #load vocab from json file
    with open(filename, 'rb') as fi:
        voc = pickle.load(fi)
    return voc


class ExtendableField(ttdata.Field):
    'A field class that allows the vocab object to be passed in' 
    #This is to avoid having to calculate the vocab every time 
    #we want to run
    def __init__(self, vocab, *args, **kwargs):
        """
        Args    
            Same args as Field except
            vocab (torchtext Vocab) : vocab to init with
                set this to None to init later

            USEFUL ARGS:
            tokenize
            fix_length (int) : max size for any example, rest are padded to this (None is defualt, means no limit)
            include_lengths (bool) : Whether to return lengths with the batch output (for packing)
        """

        super(ExtendableField, self).__init__(*args, pad_token=PAD_TOK, batch_first=True, include_lengths=True,**kwargs)
        if vocab is not None:
            self.vocab = vocab
            self.vocab_created = True
        else:
            self.vocab_created = False

    def init_vocab(self, vocab):
        if not self.vocab_created:
            self.vocab = vocab
            self.vocab_created = True

    def build_vocab(self):
        raise NotImplementedError

    def numericalize(self, arr, device=None, train=True):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (-1 or None): Device to create the Variable's Tensor on.
                Use -1 for CPU and None for the currently active GPU device.
                Default: None.
            train (boolean): Whether the batch is for a training set.
                If False, the Variable will be created with volatile=True.
                Default: True.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.LongTensor(lengths)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab, train)
        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None, train)

        arr = self.tensor_type(arr)
        if self.sequential and not self.batch_first:
            arr.t_()
        if device == -1:
            if self.sequential:
                arr = arr.contiguous()
        else:
            arr = arr.cuda(device)
            if self.include_lengths:
                lengths = lengths.cuda(device)
        #print("arr is {}".format(arr))
        if self.include_lengths:
            return arr, lengths
        return arr



class SentenceDataset(ttdata.Dataset):
    'Data set which has a single sentence per line'

    def __init__(self, path, vocab, src_seq_length=50, min_seq_length=8, n_cloze=False, add_eos=True):

        """
        Args
            path (str) : Filename of text file with dataset
            vocab (Torchtext Vocab object)
            filter_pred (callable) : Only use examples for which filter_pred(example) is TRUE
        """
        text_field = ExtendableField(vocab)

        if add_eos:
            target_field = ExtendableField(vocab, eos_token=EOS_TOK)
        else:
            target_field = ExtendableField(vocab) # added this for narrative cloze
       
        fields = [('text', text_field), ('target', target_field)]
        examples = []
        with open(path, 'r') as f:
            for line in f:
                text = line
                if n_cloze:
                    text = text.split("<TUP>") 
                    actual_event = text[-1] #last event
                    text = text[:-1] # ignore the last tuple
                    text = "<TUP>".join(text)
                    #print("cloze text is {}".format(text))
                    # text has t-1 events and target has the t event
                    examples.append(ttdata.Example.fromlist([text, actual_event], fields))
                else:
                    examples.append(ttdata.Example.fromlist([text, text], fields))

        def filter_pred(example):
            return len(example.text) <= src_seq_length and len(example.text) >= min_seq_length
 
        super(SentenceDataset, self).__init__(examples, fields, filter_pred=filter_pred)


#To Use, Create a Vocab object and and pass to a SentenceDataset object
#Create a torchtext.data.Iterator object with dset, it = torchtext.data.Iterator(dset, batch_size, sort_key=lambda x:len(x.text), sort_within_batch=True, device = -1)

class LMSentenceDataset(ttdata.Dataset):
    'Data set which has a single sentence per line'

    def __init__(self, path, vocab, src_seq_length=50, add_eos=True, n_cloze=False, min_seq_length=8): # later pass it 

        """
        everything as above though append BOS to text_field.
        """
        text_field = ExtendableField(vocab, init_token=SOS_TOK)
        if add_eos:
            target_field = ExtendableField(vocab, eos_token=EOS_TOK)
        else:
            target_field = ExtendableField(vocab) # added this for narrative cloze

        fields = [('text', text_field), ('target', target_field)]
        examples = []

        print("path is {}".format(path))
        with open(path, 'r') as f:
            for line in f:
                text = line
                #print("text is {}".format(text))
                if n_cloze:
                    text = text.split("<TUP>") 
                    actual_event = text[-1]
                    text = text[:-1] # ignore the last tuple
                    text = "<TUP>".join(text)
                    #print("cloze text is {}".format(text))
                    # text has t-1 events and target has the t event
                    examples.append(ttdata.Example.fromlist([text, actual_event], fields)) 
                else:
                    examples.append(ttdata.Example.fromlist([text, text], fields)) 

        def filter_pred(example):
            return len(example.text) <= src_seq_length and len(example.text) >= min_seq_length

        super(LMSentenceDataset, self).__init__(examples, fields, filter_pred=filter_pred)


class LMRoleSentenceDataset(ttdata.Dataset):
    'Data set which has a single sentence per line'

    def __init__(self, path, vocab, path2, vocab2, src_seq_length=50, min_seq_length=8): # later pass it 

        """
        everything as above though append BOS to text_field.
        """
        text_field = ExtendableField(vocab, init_token=SOS_TOK)
        role_field = ExtendableField(vocab2, init_token=SOS_TOK) 
        target_field = ExtendableField(vocab, eos_token=EOS_TOK)
 
        fields = [('text', text_field), ('target', target_field), ('role', role_field)]
        examples = []

        print("word path is {}".format(path))
        print("type path is {}".format(path2))

        with open(path, 'r') as f:
            with open(path2, 'r') as ft:
                for line in f:
                    text = line
                    role = ft.readline()
                    #print("text is {}".format(text))
                    #print("role is {}".format(role))
                    examples.append(ttdata.Example.fromlist([text, text, role], fields)) 

        def filter_pred(example):
            return len(example.text) <= src_seq_length and len(example.text) >= min_seq_length

        super(LMRoleSentenceDataset, self).__init__(examples, fields, filter_pred=filter_pred)


class NarrativeClozeDataset(ttdata.Dataset):
    'Data set which has a single sentence per line'

    def __init__(self, path, vocab, src_seq_length=50, min_seq_length=8, easy=False, LM=True): # later pass it 

        """
        Narrative cloze ranking based on perplexity.
        text DIST_TOK dist1 TUP_TOK dist2 .. ROLE_TOK 
        finally actual text and 4 distracted texts 
        """
       
        if LM:
            # text for LM we add SOS here.
            actual_field = ExtendableField(vocab, init_token=SOS_TOK) 
            distract1_field = ExtendableField(vocab, init_token=SOS_TOK)
            distract2_field = ExtendableField(vocab, init_token=SOS_TOK)
            distract3_field = ExtendableField(vocab, init_token=SOS_TOK)
            distract4_field = ExtendableField(vocab, init_token=SOS_TOK)
            distract5_field = ExtendableField(vocab, init_token=SOS_TOK)
            # target we add EOS here.
            actual_field_tgt = ExtendableField(vocab, eos_token=EOS_TOK) 
            distract1_field_tgt = ExtendableField(vocab, eos_token=EOS_TOK)
            distract2_field_tgt = ExtendableField(vocab, eos_token=EOS_TOK)
            distract3_field_tgt = ExtendableField(vocab, eos_token=EOS_TOK)
            distract4_field_tgt = ExtendableField(vocab, eos_token=EOS_TOK)
            distract5_field_tgt = ExtendableField(vocab, eos_token=EOS_TOK) 
        else:
            # for DAVAE SOS is added later.
            actual_field = ExtendableField(vocab) 
            distract1_field = ExtendableField(vocab)
            distract2_field = ExtendableField(vocab)
            distract3_field = ExtendableField(vocab)
            distract4_field = ExtendableField(vocab)
            distract5_field = ExtendableField(vocab)
            # target EOS is not added.
            actual_field_tgt = ExtendableField(vocab)
            distract1_field_tgt = ExtendableField(vocab)
            distract2_field_tgt = ExtendableField(vocab)
            distract3_field_tgt = ExtendableField(vocab)
            distract4_field_tgt = ExtendableField(vocab)
            distract5_field_tgt = ExtendableField(vocab)

        fields = [('actual', actual_field), ('actual_tgt', actual_field_tgt), ('dist1', distract1_field),  ('dist1_tgt', distract1_field_tgt), ('dist2', distract2_field),
                   ('dist2_tgt', distract2_field_tgt), ('dist3', distract3_field),  ('dist3_tgt', distract3_field_tgt), ('dist4', distract4_field),  ('dist4_tgt', distract4_field_tgt), 
                   ('dist5', distract5_field), ('dist5_tgt', distract5_field_tgt)]

        examples = []
        
        with open(path, 'r') as f: 
            for idx, line in enumerate(f):
                text = line.strip() 
                
                if easy: # EASY narrative cloze task. WONT ADD ROLE FOR THIS AS NOT USED ANYMORE.
                    """
                    actual, dists = text.split(DIST_TOK)
                    actual = actual.strip()
                    dists = dists.strip()
                    seed = actual.split(TUP_TOK) 
                    # data list has full sentences
                    data_list = [actual, actual]
                    # distractions 
                    seed = seed[:-1]
                    seed = [a.strip() for a in seed]
                    seed = " ".join(seed)
                   
                    for dist in dists.split(TUP_TOK): 
                        dist = "{} {} {}".format(seed, TUP_TOK, dist.strip()) 
                        data_list.extend([dist, dist])
                    """

                else: #HARD narrative cloze task
                    sents = text.split(DIST_TOK)
                    assert len(sents) == 6, "Orginal + 5 distractors" 

                    sents = [sent.strip() for sent in sents]
                    actual = sents[0].strip()
                    # CHECK 2 again
                    assert len(actual.split(TUP_TOK)) == 6, "All sentences must have 6 events."
                    dists = sents[1:]
                    seed = actual.split(TUP_TOK)[0].strip()
                    data_list = [actual, actual]
                    for dist in dists:
                        dist = "{} {} {}".format(seed, TUP_TOK, dist)
                        # CHECK 2 again.
                        assert len(dist.split(TUP_TOK)) == 6, "All sentences must have 6 events."
                        data_list.extend([dist, dist])

                
                assert len(data_list) == 12, "6 sentences: text and target so 12."
                #if idx == 1: print(data_list)
                examples.append(ttdata.Example.fromlist(data_list, fields)) 

        def filter_pred(example):
            # at this point SOS hor EOS has not been added  
            return len(example.actual) <= src_seq_length and len(example.actual) >= min_seq_length

        super(NarrativeClozeDataset, self).__init__(examples, fields, filter_pred=filter_pred)

