import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchtext.data import Iterator as BatchIter
import torch.nn.functional as F

import generate as ge
from Beam import Beam
import data_utils as du
from data_utils import EOS_TOK, SOS_TOK, PAD_TOK, TUP_TOK, MAX_EVAL_SEQ_LEN, MIN_EVAL_SEQ_LEN
from masked_cross_entropy import masked_cross_entropy
from decode_utils import transform, get_tups, get_pred_events

#from data.generate_tups import VERB, SUB, OBJ, PREP 
VERB = "__V__"
SUB = "__S__"
OBJ = "__O__"
PREP = "__P__"

def generate(use_cuda=False, device=-1):
    
    vocab = du.load_vocab(args.vocab)
    eos_id = vocab.stoi[EOS_TOK]
    pad_id = vocab.stoi[PAD_TOK]
    sos_id = vocab.stoi[SOS_TOK]
    tup_id = vocab.stoi[TUP_TOK] 

    assert False == (args.perplexity and args.seed and args.ranking), "Only 1 can be True at a time."
    # Batch size during decoding is set to 1
    assert args.batch_size == 1, "Set batch size to 1 during decoding."  

    # Load the model.
    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=lambda s, loc: s)
   
    # set the eval mode
    model.eval()
    # to decode without cuda
    model.use_cuda = False

    # TEMP FIX to work with old models without this parameter.
    #model.type_emb = None   

    # TASK SPECIFIC FUNCTION CALLS 
    if args.ranking:
        # HARD only. Easy one has been deactivated. 
        do_ranking(model, vocab) 
    elif args.perplexity: 
        get_perplexity(model, vocab)
    elif args.seed: 
        gen_from_seed(model, vocab, eos_id, pad_id, sos_id, tup_id)
    else:
        print("NOT IMPLEMENTED. RETURNING.")
        return
          


def do_ranking(model, vocab):

    dataset = du.NarrativeClozeDataset(args.data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN)
    batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.actual), train=False, device=device) 
    
    ranked_acc = 0.0
    if args.emb_type:
        print("RANKING WITH ROLE EMB")
        vocab2 = du.load_vocab(args.vocab2)
        role_dataset = du.NarrativeClozeDataset(args.role_data, vocab2, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN)
        role_batches = BatchIter(role_dataset, args.batch_size, sort_key=lambda x:len(x.actual), train=False, device=device)

        assert len(dataset) == len(role_dataset), "Dataset and Role dataset must be of same length."
       
        for iteration, (bl, rbl) in enumerate(zip(batches, role_batches)):

            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
            
            ## DATA STEPS 
            all_texts = [bl.actual, bl.actual_tgt, bl.dist1, bl.dist1_tgt, bl.dist2, bl.dist2_tgt, bl.dist3, bl.dist3_tgt,
            bl.dist4, bl.dist4_tgt, bl.dist5, bl.dist5_tgt] # each is a tup
            
            all_roles = [rbl.actual, rbl.dist1, rbl.dist2, rbl.dist3, rbl.dist4, rbl.dist5] # tgts are not needed for role  
            assert len(all_roles) == 6, "6 = 6 * 1."

            assert len(all_texts) == 12, "12 = 6 * 2."
            
            all_texts_vars = []
            all_roles_vars = []

            if use_cuda:     
                for tup in all_texts:
                    all_texts_vars.append((Variable(tup[0].cuda(), volatile=True), tup[1]))                
                for tup in all_roles:
                    all_roles_vars.append((Variable(tup[0].cuda(), volatile=True), tup[1]))

            else: 
                for tup in all_texts:
                    all_texts_vars.append((Variable(tup[0], volatile=True), tup[1]))
                for tup in all_roles:
                    all_roles_vars.append((Variable(tup[0], volatile=True), tup[1]))
 
      
            # will itetrate 2 at a time using iterator and next
            vars_iter = iter(all_texts_vars)
            roles_iter = iter(all_roles_vars)

            # run the model and collect ppls for all 6 sentences
            pps = []
            for tup in vars_iter:
                ## INIT AND DECODE before every sentence
                hidden = model.init_hidden(args.batch_size)
                next_tup = next(vars_iter) 
                role_tup = next(roles_iter)
                nll = calc_perplexity(args, model, tup[0], vocab, next_tup[0], next_tup[1], hidden, role_tup[0])  
                pp = torch.exp(nll)
                #print("NEG-LOSS {} PPL {}".format(nll.data[0], pp.data[0]))
                pps.append(pp.data.numpy()[0])

            # low perplexity == top ranked sentence- correct answer is the first one of course
            assert len(pps) == 6, "6 targets."
            #print("\n")
            all_texts_str = [transform(text[0].data.numpy()[0], vocab.itos) for text in all_texts_vars]
            #print("ALL: {}".format(all_texts_str))
            min_index = np.argmin(pps)
            if min_index == 0:
                ranked_acc += 1
                #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
                #print("CORRECT: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #else:
                # print the ones that are wrong
                #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
                #print("WRONG: {}".format(transform(all_texts_vars[min_index+2][0].data.numpy()[0], vocab.itos)))

            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break
     
        ranked_acc /= (iteration+1) * 1/100 # multiplying to get percent 
        print("Average acc(%): {}".format(ranked_acc))
        return ranked_acc 


    else: # THIS IS FOR MODEL WITHOUT ROLE EMB

        print("RANKING WITHOUT ROLE EMB.")
        for iteration, bl in enumerate(batches):

            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
            
            ## DATA STEPS 
            all_texts = [bl.actual, bl.actual_tgt, bl.dist1, bl.dist1_tgt, bl.dist2, bl.dist2_tgt, bl.dist3, bl.dist3_tgt,
            bl.dist4, bl.dist4_tgt, bl.dist5, bl.dist5_tgt] # each is a tup 

            assert len(all_texts) == 12, "12 = 6 * 2."
             
            all_texts_vars = []    
            if use_cuda: 
                for tup in all_texts:
                    all_texts_vars.append((Variable(tup[0].cuda(), volatile=True), tup[1])) 
            else:  
                for tup in all_texts:
                    all_texts_vars.append((Variable(tup[0], volatile=True), tup[1]))
                     
      
            # will itetrate 2 at a time using iterator and next
            vars_iter = iter(all_texts_vars)
            
            # run the model for all 6 sentences
            pps = []
            for tup in vars_iter:
                ## INIT AND DECODE before every sentence
                hidden = model.init_hidden(args.batch_size)
                next_tup = next(vars_iter)
           
                nll = calc_perplexity(args, model, tup[0], vocab, next_tup[0], next_tup[1], hidden) 
                pp = torch.exp(nll)
                #print("NEG-LOSS {} PPL {}".format(nll.data[0], pp.data[0]))
                pps.append(pp.data.numpy()[0])

            # low perplexity == top ranked sentence- correct answer is the first one of course
            assert len(pps) == 6, "6 targets."
            #print("\n")
            all_texts_str = [transform(text[0].data.numpy()[0], vocab.itos) for text in all_texts_vars]
            #print("ALL: {}".format(all_texts_str))
            min_index = np.argmin(pps)
            if min_index == 0:
                ranked_acc += 1
                #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
                #print("CORRECT: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
            #else:
                # print the ones that are wrong
                #print("TARGET: {}".format(transform(all_texts_vars[1][0].data.numpy()[0], vocab.itos)))
                #print("WRONG: {}".format(transform(all_texts_vars[min_index+2][0].data.numpy()[0], vocab.itos)))


            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break
     
        ranked_acc /= (iteration+1) * 1/100 # multiplying to get percent 
        print("Average acc(%): {}".format(ranked_acc))
        return ranked_acc




def gen_from_seed(model, vocab, eos_id, pad_id, sos_id, tup_id):

    if args.emb_type: # GEN FROM SEED WITH ROLE EMB
        print("GEN SEED WITH ROLE EMB")
        vocab2 = du.load_vocab(args.vocab2)
        # will use this to feed in role ids in beam decode
        ROLES = [vocab2.stoi[TUP_TOK], vocab2.stoi[VERB], vocab2.stoi[SUB], vocab2.stoi[OBJ], vocab2.stoi[PREP]]
        dataset = du.LMRoleSentenceDataset(args.data, vocab, args.role_data, vocab2, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN) #put in filter pred later
        dataset = du.LMRoleSentenceDataset(args.data, vocab, args.role_data, vocab2) #put in filter pred later
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=device) 
 
        for iteration, bl in enumerate(batches):

            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
            
            ## DATA STEPS 
            batch, batch_lens = bl.text
            target, target_lens = bl.target
            role, role_lens = bl.role
                          
            if use_cuda:  
                batch = Variable(batch.cuda(), volatile=True)
                role = Variable(role.cuda(), volatile=True)
            else: 
                batch = Variable(batch, volatile=True)
                role = Variable(role, volatile=True)
               
            ## INIT AND DECODE
            hidden = model.init_hidden(args.batch_size) 
            #run the model first on t-1 events, except last word. we know corresponding role ids as well.
            seq_len = batch.size(1) 
            for i in range(seq_len-1):
                inp = batch[:, i]
                inp = inp.unsqueeze(args.batch_size)
                typ = role[:, i]
                typ = typ.unsqueeze(1)
                _, hidden = model(inp, hidden, typ)

            #print("seq len {}, decode after {} steps".format(seq_len, i+1))
            # beam set current state to last word in the sequence
            beam_inp = batch[:, i+1]
            # do not need this anymore as assuming last sequence role obj is prep.
            #role_inp = role[:, i+1]
 #           print("ROLES LIST: {}".format(ROLES))
 #           print("FIRST ID: {}".format(role[:, i+1]))

            # init beam initializes the beam with the last sequence element. ROLE is a list of roe type ids.   
            outputs = beam_decode(model, beam_inp, hidden, args.max_len_decode, args.beam_size, pad_id, sos_id, eos_id, tup_idx=tup_id, init_beam=True, roles=ROLES)
            predicted_events = get_pred_events(outputs, vocab)  

            print("CONTEXT: {}".format(transform(batch.data.squeeze(), vocab.itos)))  
            print("PRED_t: {}".format(predicted_events)) # n_best stitched together. 

            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break


    else: 
        print("GEN SEED WITHOUT ROLE EMB")
        dataset = du.LMSentenceDataset(args.data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN) #put in filter pred later
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=device) 
        for iteration, bl in enumerate(batches):
        
            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
            
            ## DATA STEPS 
            batch, batch_lens = bl.text
            target, target_lens = bl.target 
     
            if use_cuda: 
                batch = Variable(batch.cuda(), volatile=True)                
            else: 
                batch = Variable(batch, volatile=True) 
     
            ## INIT AND DECODE
            hidden = model.init_hidden(args.batch_size) 

            #run the model first on t-1 events, except last word
            seq_len = batch.size(1) 
            for i in range(seq_len-1):
                inp = batch[:, i]
                inp = inp.unsqueeze(args.batch_size)
                _, hidden = model(inp, hidden)

            #print("seq len {}, decode after {} steps".format(seq_len, i+1))
            # beam set current state to last word in the sequence
            beam_inp = batch[:, i+1] 

            # init beam initializesthe beam with the last sequence element 
            outputs = beam_decode(model, beam_inp, hidden, args.max_len_decode, args.beam_size, pad_id, sos_id, eos_id, tup_idx=tup_id, init_beam=True)
            predicted_events = get_pred_events(outputs, vocab)  

            print("CONTEXT: {}".format(transform(batch.data.squeeze(), vocab.itos))) 
            print("PRED_t: {}".format(predicted_events)) # n_best stitched together. 

            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break 

   

def get_perplexity_avg_line(model, vocab):    
    total_loss = 0.0
    if args.emb_type: # GET PERPLEXITY WITH ROLE EMB
        print("PERPLEXITY WITH ROLE EMB")
        vocab2 = du.load_vocab(args.vocab2)
        dataset = du.LMRoleSentenceDataset(args.data, vocab, args.role_data, vocab2, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN) #put in filter pred later
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=device) 

        print("DATASET {}".format(len(dataset)))
        for iteration, bl in enumerate(batches):
            
            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
                
            ## DATA STEPS 
            batch, batch_lens = bl.text
            target, target_lens = bl.target         
            role, role_lens = bl.role

            if use_cuda: 
                batch = Variable(batch.cuda(), volatile=True) 
                target = Variable(target.cuda(), volatile=True)
                role = Variable(role.cuda(), volatile=True)
            else: 
                batch = Variable(batch, volatile=True) 
                target = Variable(target, volatile=True)
                role = Variable(role, volatile=True)
                 
            ## INIT AND DECODE
            hidden = model.init_hidden(args.batch_size)                  
            ce_loss = calc_perplexity(args, model, batch, vocab, target, target_lens, hidden, role)
            #print("Loss {}".format(ce_loss))
            total_loss = total_loss + ce_loss.data[0]
            
            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break

        # after iterating over all examples 
        loss = total_loss / (iteration+1)
        print("Average Loss: {}".format(loss))
        return loss

    else: 
        print("PERPLEXITY WITHOUT ROLE EMB")
        dataset = du.LMSentenceDataset(args.data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN) #put in filter pred later
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=device)
        for iteration, bl in enumerate(batches):
         
            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
            
            ## DATA STEPS 
            batch, batch_lens = bl.text
            target, target_lens = bl.target         
     
            if use_cuda: 
                batch = Variable(batch.cuda(), volatile=True) 
                target = Variable(target, volatile=True) 
            else: 
                batch = Variable(batch, volatile=True) 
                target = Variable(target, volatile=True)
                  
            ## INIT AND DECODE
            hidden = model.init_hidden(args.batch_size)  
            ce_loss = calc_perplexity(args, model, batch, vocab, target, target_lens, hidden)
            #print("Loss {}".format(ce_loss))
            total_loss = total_loss + ce_loss.data[0]

            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break


        # after iterating over all examples 
        loss = total_loss / (iteration+1)
        print("Average Loss: {}".format(loss))
        return loss 


def get_perplexity(model, vocab):    
    total_loss = 0.0
    total_words = 0
    if args.emb_type: # GET PERPLEXITY WITH ROLE EMB
        print("PERPLEXITY WITH ROLE EMB")
        vocab2 = du.load_vocab(args.vocab2)
        dataset = du.LMRoleSentenceDataset(args.data, vocab, args.role_data, vocab2, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN) #put in filter pred later
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=device) 

        print("DATASET {}".format(len(dataset)))
        for iteration, bl in enumerate(batches):
            
            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
                
            ## DATA STEPS 
            batch, batch_lens = bl.text
            target, target_lens = bl.target         
            role, role_lens = bl.role

            if use_cuda: 
                batch = Variable(batch.cuda(), volatile=True) 
                target = Variable(target.cuda(), volatile=True)
                role = Variable(role.cuda(), volatile=True)
            else: 
                batch = Variable(batch, volatile=True) 
                target = Variable(target, volatile=True)
                role = Variable(role, volatile=True)
                 
            ## INIT AND DECODE
            hidden = model.init_hidden(args.batch_size)                  
            ce_loss = calc_perplexity(args, model, batch, vocab, target, target_lens, hidden, role)
            #print("Loss {}".format(ce_loss))
            total_loss = total_loss + ce_loss.data[0]*target_lens.float().sum()

            total_words += target_lens.sum()
            
            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break

        # after iterating over all examples 
        loss = total_loss / total_words.float()
        print("Average Loss: {}".format(loss))
        return loss

    else: 
        print("PERPLEXITY WITHOUT ROLE EMB")
        dataset = du.LMSentenceDataset(args.data, vocab, src_seq_length=MAX_EVAL_SEQ_LEN, min_seq_length=MIN_EVAL_SEQ_LEN) #put in filter pred later
        batches = BatchIter(dataset, args.batch_size, sort_key=lambda x:len(x.text), train=False, device=device)
        for iteration, bl in enumerate(batches):
         
            if (iteration+1)%25 == 0:
                print("iteration {}".format(iteration+1))
            
            ## DATA STEPS 
            batch, batch_lens = bl.text
            target, target_lens = bl.target         
     
            if use_cuda: 
                batch = Variable(batch.cuda(), volatile=True) 
                target = Variable(target, volatile=True) 
            else: 
                batch = Variable(batch, volatile=True) 
                target = Variable(target, volatile=True)
                  
            ## INIT AND DECODE
            hidden = model.init_hidden(args.batch_size)  
            ce_loss = calc_perplexity(args, model, batch, vocab, target, target_lens, hidden)
            #print("Loss {}".format(ce_loss))
            total_loss = total_loss + ce_loss.data[0]*target_lens.float().sum()

            total_words += target_lens.sum()

            if (iteration+1) == args.max_decode:
                print("Max decode reached. Exiting.")
                break


        # after iterating over all examples 
        loss = total_loss / total_words.float()
        print("Average Loss: {}".format(loss))
        return loss 


def calc_perplexity(args, model, batch, vocab, target, target_lens, hidden, role=None):
   
    logits = []
    for i in range(batch.size(1)): #decode input
        inp = batch[:, i] 
        inp = inp.unsqueeze(1)
        if args.emb_type:
            typ = role[:, i]
            typ = typ.unsqueeze(1)
            logit, hidden = model(inp, hidden, typ) # add in the role here
        else:
            # keep updating the hidden state accordingly
            logit, hidden = model(inp, hidden) 
        logits += [logit]

    logits = torch.stack(logits, dim=1)
    # loss for the sequence
    # making changes to not consider the EOS when calculating the scores.
    target_lens = target_lens - 1
    loss = masked_cross_entropy(logits, target, Variable(target_lens)) 
    return loss

           

def beam_decode(model, input, hidden, max_len_decode, beam_size, pad_id, sos_id, eos_id, tup_idx=4, batch_size=1, use_constraints=True, init_beam=False, roles=None):
    # hidden [1, 1, hidden_size] 

    assert beam_size > 0 and batch_size == 1, "Beam decoding batch size must be 1 and Beam size greater than 0." 

    # Helper functions for working with beams and batches
    def var(a): return Variable(a, volatile=True) 

    def bottle(m):
        return m.view(batch_size * beam_size, -1)

    def unbottle(m):
        return m.view(beam_size, batch_size, -1)

    def beam_update(e, idx, positions, beam_size):  
        sizes = e.size() # [1, beam_size, hidden_size] 
        br = sizes[1] 
        if len(sizes) == 3:
            sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                 sizes[2])[:, :, idx]
        else:
            sent_states = e.view(sizes[0], beam_size,
                                 br // beam_size,
                                 sizes[2],
                                 sizes[3])[:, :, idx]

        # [1, beam_size, hidden_size]
        indexed_before = sent_states.data.index_select(1, positions)  
        sent_states.data.copy_(
            sent_states.data.index_select(1, positions))  
        indexed_after = sent_states.data.index_select(1, positions) 
         

    # 1 beam object as we have batch_size 1 during decoding
    beam = [Beam(beam_size, n_best=args.n_best,
                cuda=use_cuda,
                pad=pad_id,
                eos=eos_id,
                bos=sos_id,
                min_length=10)]     

    if init_beam:
        # id of last element in seq to init the beam  
        for b in beam:
            b.next_ys[0][0] = np.asscalar(input.data.numpy()[0])
     
    # [1, beam_size, hidden_size]
    hidden = hidden.repeat(1, beam_size, 1)  

    # this comes from the known role id of the last seqence object.
    #if args.emb_type:
        #inp2 = role.repeat(1, beam_size)

    verb_list = [[]]*beam_size #for constraints
    # run the decoder to generate the sequence 
    for i in range(max_len_decode): 

        # one all beams have EOS break 
        if all((b.done() for b in beam)):
            break

        # No need to explicitly set the input to previous output - beam advance does it. Make sure.
        inp = var(torch.stack([b.get_current_state() for b in beam])
                              .t().contiguous().view(-1)) #[beam_size]

        # Tested that the last output is the input in the next time step. 
        # Run one step of the decoder
        # dec_out: beam x rnn_size
        inp = inp.unsqueeze(1)

        if args.emb_type:
            curr_idx = i%5
            # this gives the index of the role type: [tup, v, s, o, prep]
            curr_role = roles[curr_idx] 
            # wrap into a tensor and make a var. repeat beam times
            inp2 = var(torch.LongTensor([curr_role])).repeat(beam_size, 1)
            logit, hidden = model(inp, hidden, inp2)
        else:
            logit, hidden = model(inp, hidden)
        
        # [1, beam_size, hidden_size]
        logit = torch.unsqueeze(logit, 0)  
        probs = F.log_softmax(logit, dim=2).data  
        out = unbottle(probs) # [beam_size, 1, vocab_size]
        out.log()
        
        # Advance each beam. We have 1 beam object. 
        for j, b in enumerate(beam): 
            #print("OUT: {}".format(out[:, j])) # [beam_size, vocab_size]
            if use_constraints:
                b.advance(ge.schema_constraint(out[:, j], b.next_ys[-1], verb_list)) 
            else:
                b.advance(out[:, j]) 
                
            beam_update(hidden, j, b.get_current_origin(), beam_size)
            if use_constraints:
                verb_list = ge.update_verb_list(verb_list, b, tup_idx) 

             
    # extract sentences from beam and return
    ret = _from_beam(beam, args.n_best)
    return ret

        
def _from_beam(beam, n_best=1):
    ret = {"predictions": [],
           "scores": []}
    for b in beam: # Only 1 beam object. 
        scores, ks = b.sort_finished(minimum=n_best)
        hyps = []
        for i, (times, k) in enumerate(ks[:n_best]):
            hyp = b.get_hyp(times, k)
            hyps.append(hyp)
        ret["predictions"].append(hyps)
        ret["scores"].append(scores)   
    
    return ret['predictions']

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, help='location of the data corpus')
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--model', type=str, help='model checkpoint to use') 
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--random_seed', type=int, default=11, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')     
    parser.add_argument('-max_len_decode', type=int, default=50, help='Maximum prediction length.') 
    # beam related
    parser.add_argument('--beam_size',  type=int, default=-1, help='Beam size')
    parser.add_argument('--n_best', type=int, default=1, help="""outputs the n_best decoded sentences""")
    # types of evaluation 
    # THESE NARRATIE CLOZE TASKS ARE NOT USED ANYMORE
    #parser.add_argument('--n_cloze', action='store_true', help="""narrative cloze recall based evaluation""")
    #parser.add_argument('--easy', action='store_true', help="""narrative cloze ranking: easy. predict last event.""")
    parser.add_argument('--perplexity',  action='store_true')
    parser.add_argument('--ranking', action='store_true', help="""narrative cloze ranking""") 
    parser.add_argument('--seed', action='store_true', help="""seed based testing""")
    # max decoded
    parser.add_argument('--max_decode', type=int, default=1000000, help="""max sentences to be evaluated/decoded.""") 
    # role data related
    parser.add_argument('--emb_type', action='store_true')
    parser.add_argument('--role_data', type=str, help='location of the role data corpus')
    parser.add_argument('--vocab2', type=str, help='location of role vocab')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.cuda and torch.cuda.is_available():
        device = 0
        use_cuda = True
    elif args.cuda and not torch.cuda.is_available():
        print("You do not have CUDA, turning cuda off")
        device = -1 
        use_cuda = False
    else:
        device = -1
        use_cuda=False 

    generate(use_cuda=use_cuda, device=device)
