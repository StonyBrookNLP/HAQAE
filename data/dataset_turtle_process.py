############################################
#   Dataset object, for efficiently streaming the
#   SVO triple data into a TensorFlow
#   feed
############################################
import os
import string
import itertools
from collections import namedtuple
common_verbs = ['be', 'were', 'been', 'is', "'s", 'have', 'had', 'do', 'did', 'done', 'say', 'said', 'go', 'went', 'gone', 'get', 'got', 'gotton', 'told']
auxilary_verbs = ['be', 'am', 'are', 'is', 'was', 'were', 'being', 'been', 'can', 'could', 'dare', 'do', 'does', 'did', 'have', 'has', 'had', 'having', 'may', 'might', 'must', 'ought', 'shall', 'should', 'will', 'would', "'s"]
prepositions = ['of', 'to', 'in', 'for', 'on', 'by', 'at', 'with', 'from', 'into', 'during', 'including', 'until', 'about', 'like', 'through', 'over', 'before', 'after', 'around', 'near', 'above', 'without', 'but', 'up', 'down']
SENT_SEP="|SENT|"
TUP_SEP="|TUP|"
NO_PREP = "_NULL_"

common_nouns = ['i', 'we', 'it', 'them', 'you']

SimpleTuple = namedtuple('SimpleTuple', 'sub verb obj prep')

translator = str.maketrans('','',string.punctuation)

class Sentence:
    def __init__(self, sentid, text, tuples):
        """
        sentid is an integer id, text is the sentence text itself, and tuples
        is a list of SimpleTuples containing the tuples found in the sentence
        """
        self.sent_id=sentid
        self.text=text
        self.tuples=tuples

    def __str__(self):
        string = "%s:%s\n" % (self.sent_id, self.text)
        for i in self.tuples:
            string+="\t(%s, %s, %s)\n" % (i.sub, i.verb, i.obj)
        return string
 
class Document:
    def __init__(self, doc_id, sentences):
        """
        sentences is a list of Sentence objects
        """
        self.doc_id=doc_id
        self.sentences=sentences

    def __str__(self):
        string = "***********%s************" % self.doc_id
        for i in self.sentences:
            string += str(i)
        return string


def process_prep(v):
    """process prepositions"""
    splits = v.split()
    if len(splits) == 1:
        return (v, NO_PREP)
    elif len(splits) == 2 and splits[1] in prepositions:
        return (splits[0], splits[1])
    else:
        return None
        
def crap_tuple(splits):
    """Return true if this tuple is a bit crap"""
    s = splits[0].strip().lower()
    v = splits[1].strip().lower()
    o = splits[2].strip().lower()

    if 'said' in v  \
        or v == "" \
        or v in common_verbs \
        or v in prepositions \
        or v[-3:] == ' as' \
        or v[-3:] == 'as ' \
        or len(v.split()) >= 3 \
        or s == o:

        return True

    elif "'" in s \
        or s == "" \
        or s in common_nouns:

        return True

    elif "'" in o \
        or o == "" \
        or o in common_nouns:

        return True
    else:
        return False


class DocOpenIE_Dataset: ##USE THISSSSSSS
    'For the document level version of the openie dataset, where each line stores a document, yield Document objects' 
    #Also the iterator in this class removes some of the redundent expressions ollie extracts
    def __init__(self,filename):
        self.filename=filename

    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as fi:
            for doc in fi: #each line is a document
                doc_sents = doc.split(SENT_SEP)
                doc_id = ""
                sents_list = []
                for sents in doc_sents:
                    sent_tups = sents.split(TUP_SEP)
                    if len(sent_tups) >= 2:
                        info = sent_tups[0].split("|")
                        tuples = sent_tups[1:]
                        if len(info) == 3: 
                            doc_id=info[0]
                            sent_id=info[1]
                            sent_text=info[2]
                            tuples_list = []
                            prev_verbs = []
                            for tups in tuples:
                                splits=tups.split("|")
                                if len(splits) < 3:
                                    continue
                                verb = splits[1].strip().lower()
                                vlist = [x for x in verb.split() if x not in auxilary_verbs] #remove the auxilariry verbs (so remember, during test time we should also remove verbs)
                                if vlist:
                                    v = " ".join(vlist)
                                    v = v.translate(translator)
                                    main_verb = " ".join([x for x in vlist if x not in prepositions])
                                    v_prep = process_prep(v)
                                    if not any([main_verb in x for x in prev_verbs]) and not crap_tuple((splits[0], v, splits[2])) and v_prep is not None: #make sure this tuple doesnt have the same main verb as any previous ones 
                                        sub=splits[0].strip().lower()
                                        if len(sub.split()) > 1:
                                            sub = "_".join(sub.split())
                                        verb=v_prep[0]
                                        if len(verb.split()) > 1:
                                            verb = "_".join(verb.split())
                                        obj=splits[2].strip().lower()
                                        if len(obj.split()) > 1:
                                            obj = "_".join(obj.split())
                                        prep = v_prep[1]
                                        if len(prep.split()) > 1:
                                            prep = "_".join(prep.split())

                                        tuples_list.append(SimpleTuple(sub=sub,
                                                           verb=verb, 
                                                           obj=obj,
                                                           prep=prep))  #add tuple to this sentence, keep prepositions with verb
                                        prev_verbs.append(main_verb)   #since this verb was used, add it to list of previously used main verbs

                                    #tuples_list.append(SimpleTuple(sub=splits[0], verb=v, obj=splits[2])) #uncomment this and comment the above 4 lines if using regular not openie
                            if tuples_list:
                                sents_list.append(Sentence(sent_id, sent_text, tuples_list))
                        else:
                            print("Info parsing error")
                    else:
                        print("sent tups parsing error")
                if sents_list:
                    yield Document(doc_id, sents_list)
                    

def get_stopwords(filename="stopwords.txt"):
    stopwords = []
    with open(filename, 'r') as fi:
        for line in fi:
            stopwords.append(line.strip())
    return stopwords



