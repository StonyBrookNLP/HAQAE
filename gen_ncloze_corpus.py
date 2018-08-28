### this will read the actual val file and create Narrative cloze corpus
### 1. EASY TEST [t-1 events] and predict the t th event. Output each line as: actual sentence (TUP_TOK for each event/tuple) DIST_TOK dist1 (1 event/tuple) TUP_TOK dist2 ... With and without entity repalcement. 
### 2. HARD TEST 1 st event given and predict the next t events. t >=3. <first_event> DIST_TOK <dist1> DIST_TOK <dist2>. each dist has tups seperated by TUP_TOK 

import sys
import random

seed = 11
random.seed(seed)

from data_utils import TUP_TOK, DIST_TOK 

EASY = 0
HARD = 1

###
#an event sequence should be 6: 1 seed event + 5 following events. All sentences must be of same length.
###

class NClozeSampler():
    def __init__(self, fin, fout, num, finrole):
        self.num = num
        # actual sentences
        self.texts = []
        # last event for each sentences (EASY) or last 3 or more events (HARD)
        self.events = []
        # role file indexes
        self.role_idxs = []
        self.roles = []
        # check
        assert EASY == False, "Easy should be deactivated now."

        # 1. read the file and store in lists
        counter = 0
        with open(fin, "r") as fread:

            for l_idx, line in enumerate(fread):  
                if counter >= self.num:
                    print("Valid num reached. Done reading.")
                    break

                text = line.strip()

                if EASY:
                    """
                    event = text.split(TUP_TOK)[-1].strip()

                     # EASY CASE only put valid t event (V, S, O, P) 
                    if len(event.split()) == 4:
                        self.texts.append(text) 
                        self.events.append(event) # list of strings
                        counter += 1
                    else:
                        print("NOPES {}:{}".format(len(event.split()), event))
                    """

                else:
                    # CHECK 1 on the first event.
                    first = text.split(TUP_TOK)[0].strip()
                    if len(first.split()) != 4:
                        continue # go to the next

                    events = text.split(TUP_TOK)[1:]
                     
                    if type(events) == list:
                        # CHECK 2 collect ones with 5 events except the seed event
                        if len(events) == 5: 
                            # CHECK 1 on subsequent events
                            if all([len(event.split())==4 for event in events]): 
                                self.texts.append(text) 
                                self.events.append(events) # this is a list of lists 
                                if finrole is not None:
                                    # keep track of which lines read
                                    self.role_idxs.append(l_idx)
                                counter += 1
 
                   
        assert len(self.texts) == len(self.events), "num texts and events must be same."
        print("READ {}/{} data. Moving to generation.".format(len(self.texts), self.num))

        # collect the repective role fields
        if finrole is not None:
            with open(finrole, "r") as frread:
                for r_idx, role in enumerate(frread):
                    if r_idx in self.role_idxs:
                        self.roles.append(role.strip())


        # 2. Go over stored lists and generate dataset then write to file
        if finrole is not None:
            with open(fout2, "w") as frwrite:
                with open(fout, "w") as fwrite:
                    for idx, text in enumerate(self.texts):

                        # sampling wihout replacement: just to be safe
                        random_nums = list(range(0, len(self.events)))
                        del random_nums[idx]
                        dist_nums = random.sample(random_nums, 5) 
                        dists = [self.events[dist] for dist in dist_nums]

                        if EASY:
                            """
                            actual_event = self.events[idx].strip().split()
                            assert len(actual_event) == 4, "Error: filtered tuples must have 4 entries."
           
                            # Entity replacement
                            dists = [dist.strip().split() for dist in dists]
                            dists = [[dist[0], actual_event[1], actual_event[2], dist[3]] for dist in dists] 
                            dists = [" ".join(dist) for dist in dists]
                            dist1, dist2, dist3, dist4, dist5 = dists 

                            # write to file 
                            fwrite.write("{} {} {} {} {} {} {} {} {} {} {}\n".format( \
                                text, DIST_TOK, dist1, TUP_TOK, dist2, TUP_TOK, dist3, TUP_TOK, dist4, TUP_TOK, dist5))
                            """

                        # No entity replacement for this now.
                        if HARD:
                            actual_events = self.events[idx] # later can be use for entity replacement
                            #actual_events = TUP_TOK.join(actual_events)

                            # role based stuff 
                            role_text = self.roles[idx] 
                            role_dists = [self.roles[idx] for idx in dist_nums]
                            role_dists = [role_dist.split(TUP_TOK) for role_dist in role_dists] # list of lists
                            role_dists = [role_dist[1:] for role_dist in role_dists] #ignore the first event and keep the rest
                            role_dists = [TUP_TOK.join(role_dist) for role_dist in role_dists]
                            role_dists = [role_dist.strip() for role_dist in role_dists]


                            # join the lists inside
                            dists = [TUP_TOK.join(dist) for dist in dists]
                            dists = [dist.strip() for dist in dists] 
                            dist1, dist2, dist3, dist4, dist5 = dists

                            # write words to file 
                            fwrite.write("{} {} {} {} {} {} {} {} {} {} {}\n".format( \
                            text, DIST_TOK, dist1, DIST_TOK, dist2, DIST_TOK, dist3, DIST_TOK, dist4, DIST_TOK, dist5))

                            # write roles to file 
                            r_dist1, r_dist2, r_dist3, r_dist4, r_dist5 = role_dists 
                            frwrite.write("{} {} {} {} {} {} {} {} {} {} {}\n".format( \
                            role_text, DIST_TOK, r_dist1, DIST_TOK, r_dist2, DIST_TOK, r_dist3, DIST_TOK, r_dist4, DIST_TOK, r_dist5))
                            
                           
        else:

            with open(fout, "w") as fwrite:
                for idx, text in enumerate(self.texts):

                    # sampling wihout replacement: just to be safe
                    random_nums = list(range(0, len(self.events)))
                    del random_nums[idx]
                    dist_nums = random.sample(random_nums, 5) 
                    dists = [self.events[dist] for dist in dist_nums]

                    if EASY:
                        """
                        actual_event = self.events[idx].strip().split()
                        assert len(actual_event) == 4, "Error: filtered tuples must have 4 entries."
       
                        # Entity replacement
                        dists = [dist.strip().split() for dist in dists]
                        dists = [[dist[0], actual_event[1], actual_event[2], dist[3]] for dist in dists] 
                        dists = [" ".join(dist) for dist in dists]
                        dist1, dist2, dist3, dist4, dist5 = dists 

                        # write to file 
                        fwrite.write("{} {} {} {} {} {} {} {} {} {} {}\n".format( \
                            text, DIST_TOK, dist1, TUP_TOK, dist2, TUP_TOK, dist3, TUP_TOK, dist4, TUP_TOK, dist5))
                        """
                        
                    # No entity replacement for this now.
                    if HARD:
                        actual_events = self.events[idx] # later can be use for entity replacement
                        #actual_events = TUP_TOK.join(actual_events)

                        # join the lists inside
                        dists = [TUP_TOK.join(dist) for dist in dists]
                        dists = [dist.strip() for dist in dists] 
                        dist1, dist2, dist3, dist4, dist5 = dists 

                        # write to file 
                        fwrite.write("{} {} {} {} {} {} {} {} {} {} {}\n".format( \
                        text, DIST_TOK, dist1, DIST_TOK, dist2, DIST_TOK, dist3, DIST_TOK, dist4, DIST_TOK, dist5))
                    

        print("File saved.")

            

if __name__ == "__main__":

    if HARD:
        print("HARD in ON.")
    else:
        print("EASY is ON.")

    # read in the original data file
    fin = sys.argv[1]
    # write the NC data in a new file
    fout = sys.argv[2]
    num = int(sys.argv[3])

    if len(sys.argv) > 4:
        print("Role file will be generated as well.")
        fin2 = sys.argv[4]
        fout2 = sys.argv[5]

        NClozeSampler(fin, fout, num, finrole=fin2)
    else:
        NClozeSampler(fin, fout, num, finrole=None)



