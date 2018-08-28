######
## Uses DocOpenIE_Dataset iterator to write triples (tups) to file. Tuples in a chunk in a line.
######
import sys
#import the required class
#from utils.dataset import DocOpenIE_Dataset
from dataset_turtle_process import DocOpenIE_Dataset

VERB = "__V__"
SUB = "__S__"
OBJ = "__O__"
PREP = "__P__"


if __name__ == "__main__":

    fread = sys.argv[1] # file to read from
    fwrite = sys.argv[2] # file to write to
    ftypew = sys.argv[3] # file towrite type to
    k = 4 #sys.argv[3] # sentence chunk size, how many sentences per data instance
    # check directory

    # collect statistics
    avg_tups = 0
    num_sentences = 0
    num_chunks = 0

    first = True
    doc_iter = DocOpenIE_Dataset(fread)

    with open(ftypew, "w") as ftout:
        with open(fwrite, "w") as fout:
            for d_id, doc in enumerate(doc_iter):
                #if d_id == 100:
                    #break
                #print("Doc id {}".format(d_id))
                #doc_tups_flag = 0
                # num sentences in the corpus
                num_sentences += len(doc.sentences) 
                for s_id, s in enumerate(doc.sentences):
            # num tuples
                    avg_tups += len(s.tuples)

                    for tup in s.tuples:
                        if first:
                            
                            word_str = "{} {} {} {} ".format(tup.verb, tup.sub, tup.obj, tup.prep) 
                            fout.write(word_str)
                            
                            ## added to get rid of data issues
                            sub=tup.sub
                            if len(sub.split()) >= 1:
                                sub_t = [SUB] * len(sub.split())
                                sub_t = " ".join(sub_t)
                            else:
                                print("SUB is NULL for doc {}.".format(d_id))
                            verb=tup.verb
                            if len(verb.split()) >= 1:
                                verb_t = [VERB] * len(verb.split())
                                verb_t = " ".join(verb_t)
                            else:
                                print("VERB is NULL for doc {}.".format(d_id))
                            obj=tup.obj
                            if len(obj.split()) >= 1:
                                obj_t = [OBJ] * len(obj.split())
                                obj_t = " ".join(obj_t)
                            else:
                                print("OBJ is NULL for doc {}.".format(d_id))
                            prep = tup.prep
                            if len(prep.split()) >= 1:
                                prep_t = [PREP] * len(prep.split())
                                prep_t = " ".join(prep_t)
                            else:
                                print("PREP is NULL for doc {}.".format(d_id))
                            ##

                            #ftout.write("{} {} {} {} ".format(VERB, SUB, OBJ, PREP))
                            type_str = "{} {} {} {} ".format(verb_t, sub_t, obj_t, prep_t)
                            ftout.write(type_str)

                            assert len(word_str.split()) == len(type_str.split()), "length should be same. {} ::: {}".format(word_str, type_str)
                            first = False
                        else:
                            
                            word_str = "<TUP> {} {} {} {} ".format(tup.verb, tup.sub, tup.obj, tup.prep) 
                            fout.write(word_str) 

                             ## added to get rid of data issues
                            sub=tup.sub
                            if len(sub.split()) >= 1:
                                sub_t = [SUB] * len(sub.split())
                                sub_t = " ".join(sub_t)
                            else:
                                print("SUB is NULL for doc {}.".format(d_id))
                            verb=tup.verb
                            if len(verb.split()) >= 1:
                                verb_t = [VERB] * len(verb.split())
                                verb_t = " ".join(verb_t)
                            else:
                                print("VERB is NULL for doc {}.".format(d_id))
                            obj=tup.obj
                            if len(obj.split()) >= 1:
                                obj_t = [OBJ] * len(obj.split())
                                obj_t = " ".join(obj_t)
                            else:
                                print("OBJ is NULL for doc {}.".format(d_id))
                            prep = tup.prep
                            if len(prep.split()) >= 1:
                                prep_t = [PREP] * len(prep.split())
                                prep_t = " ".join(prep_t)
                            else:
                                print("PREP is NULL for doc {}.".format(d_id))
                            ##
                            
                            #ftout.write("{} {} {} {} ".format(VERB, SUB, OBJ, PREP))
                            type_str = "<TUP> {} {} {} {} ".format(verb_t, sub_t, obj_t, prep_t)
                            ftout.write(type_str)

                            assert len(word_str.split()) == len(type_str.split()), "length should be same {} ::: {}".format(word_str, type_str)

                            #ftout.write("<TUP> {} {} {} {} ".format(VERB, SUB, OBJ, PREP))
                        #doc_tups_flag = 1                   
                    # written k sentences tups then add a linebreak 
                    if s_id%k == -1%k: #rather than 0 for the first comparision
                        first = True
                        num_chunks += 1
                        #print("ADDED EOS")
                        fout.write("\n".format()) #<EOS>
                        ftout.write("\n".format()) #<EOS>

                first = True
                #if not doc_tups_flag:
                    #print("Doc {} has no tups.".format(d_id)) 
                if len(doc.sentences)%k != 0: # and len(doc.sentences) != 0: # and doc_tups_flag:
                     num_chunks += 1
                     #print("ADDED EOS")
                     fout.write("\n") #<EOS>
                     ftout.write("\n") #<EOS>
        # doc level average
        #avg_tups = avg_tups/(d_id + 1)
        # sentence level average
        #avg_tups = avg_tups/num_sentences
	# chunk level average
        avg_tups = avg_tups/num_chunks
        print("Total docs {} chunks {} sentences {}".format(d_id+1, num_chunks, num_sentences))
        print("Average tups per chunk: {}".format(avg_tups))



