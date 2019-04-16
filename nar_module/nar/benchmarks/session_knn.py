'''
Adapted from: 
Jannach D, Ludewig M. When recurrent neural networks meet the neighborhood for session-based recommendation. InProceedings of the Eleventh ACM Conference on Recommender Systems 2017 Aug 27 (pp. 306-310). ACM.
SkNN (cknn.py, available in http://bit.ly/2nfNldD): 

Michael Jugovac and Dietmar Jannach and Mozhgan Karimi. "StreamingRec: A Framework for Benchmarking Stream-based News Recommenders". RecSys ’18, October 2–7, 2018. Available in http://ls13-www.cs.tu-dortmund.de/homepage/publications/jannach/Conference_RECSYS17.pdf 
V-SkNN (fixed) (KNearestNeighbor.java, available in https://github.com/mjugo/StreamingRec)

Malte Ludewig and Dietmar Jannach. 2018. Evaluation of Session-Based Recommenation
Algorithms. (2018). arXiv:1803.09587 [cs.IR] https://arxiv.org/abs/1803.09587
S-SkNN(scknn.py) and SF-SKNN (sfcknn.py), available in https://www.dropbox.com/sh/7qdquluflk032ot/AACoz2Go49q1mTpXYGe0gaANa?dl=0): ] 

if SessionBasedKNNRecommender.first_session_clicks_decay == 'same':
    SkNN
elif SessionBasedKNNRecommender.first_session_clicks_decay == 'div':
    V-SkNN
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from math import sqrt

from .benchmarks import BenchmarkRecommender
from collections import namedtuple, defaultdict

class SessionBasedKNNRecommender(BenchmarkRecommender):
    
    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        #super(Instructor, self).__init__(name, year) #Python 2
        super().__init__(clicked_items_state, params, eval_streaming_metrics)


        self.sessions_buffer_size = params['sessions_buffer_size'] #Buffer size for last processed sessions
        self.candidate_sessions_sample_size = params['candidate_sessions_sample_size'] #Number of candidate near sessions to sample
        self.sampling_strategy= params['sampling_strategy']  #(recent,random)
        self.nearest_neighbor_session_for_scoring = params['nearest_neighbor_session_for_scoring'] #Nearest neighbors to compute scores        
        
        self.similarity = params['similarity'] #(jaccard, cosine)

        #Decays weight of first user clicks in active session when finding neighbor sessions (same, div, linear, log, quadradic)
        self.first_session_clicks_decay = params['first_session_clicks_decay'] 
        self.first_session_clicks_decay_fn = getattr(self, '{}_pos_decay'.format(self.first_session_clicks_decay))
        
        #Registering a state for this recommender which persists over TF Estimator train/eval loops
        if not self.get_clf_suffix() in clicked_items_state.benchmarks_states:
            clicked_items_state.benchmarks_states[self.get_clf_suffix()] = \
                                dict({'last_sessions_buffer': [],
                                      'item_session_map': defaultdict(set)})
  
        state = clicked_items_state.benchmarks_states[self.get_clf_suffix()]
        self.last_sessions_buffer = state['last_sessions_buffer'] # Tuple: (session_id (int), item_ids (list of ints))
        self.item_session_map = state['item_session_map'] #Dict: item -> list of session_ids

        self.SessionStruct = namedtuple('SessionStruct', ['session_id', 'item_ids'])

    def get_clf_suffix(self):
        return 'sknn' if self.first_session_clicks_decay == 'same' else 'v-sknn'
        
    def get_description(self):
        return 'Session-KNN: Retrieves items from similar sessions (kNN) from last sessions buffer'
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        sessions_all_items = np.hstack([sessions_items, sessions_next_items])
        session_items_sets = list([set(filter(lambda x: x != 0, session_items)) \
                                   for session_items in sessions_all_items])
        #Add sessions to buffer and removes older ones
        self.add_sessions_to_buffer(sessions_ids, session_items_sets)   


    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):
        session_predictions = np.zeros(dtype=np.int64,
                                         shape=[sessions_items.shape[0],
                                                sessions_items.shape[1],
                                                topk])


        for row_idx, session_items in enumerate(sessions_items):

            for col_idx, item in enumerate(session_items):
                if item != 0:
                    partial_session_items = session_items[:col_idx+1]
                    #Get a sample of k-nearest neighbor sessions  
                    #based on the session clicks up to this point                    
                    neighbor_sessions_similarity = self.find_neighbors(partial_session_items)
                    item_scores = self.score_items(neighbor_sessions_similarity)
                    items_ids_ranked = list(map(lambda y: y[0], sorted(item_scores.items(), reverse=True, key=lambda x: x[1])))
                    session_predictions[row_idx, col_idx] = list(self._get_top_n_valid_items(items_ids_ranked, topk, valid_items[row_idx, col_idx]))

        return session_predictions 

    #Add sessions to buffer and removes older ones
    def add_sessions_to_buffer(self, sessions_ids, sessions_items_sets):
        new_sessions = list([self.SessionStruct(session_id=session_id, 
                                                item_ids=items_set) \
                             for session_id, items_set in zip(sessions_ids, sessions_items_sets)])
        self.last_sessions_buffer.extend(new_sessions)

        #Create a map from item_id to session ids where this item was clicked
        #to speed KNN search
        for session_id, session_items in zip(sessions_ids, sessions_items_sets):
            for item_id in session_items:
                self.item_session_map[item_id].add(session_id)        

        #Keeping buffer on specified size
        while len(self.last_sessions_buffer) > self.sessions_buffer_size:
            session_idx_to_remove = 0
            session_to_remove = self.last_sessions_buffer[session_idx_to_remove]
            for item_id in session_to_remove.item_ids:
                self.item_session_map[item_id].discard(session_to_remove.session_id)
            del self.last_sessions_buffer[session_idx_to_remove]

    def find_session_on_buffer(self, session_id):        
        """
        Binary search (leftmost value) on the first element of list of tuples (session_id)
        """
        arr = self.last_sessions_buffer

        left = 0
        right = len(arr)
        while left < right:
            mid = (left + right) // 2
            if session_id > arr[mid].session_id:
                left = mid + 1
            else:
                right = mid
        if left != len(arr) and arr[left].session_id == session_id:
            return left
        else:
            return -1

    def get_session_items_from_buffer(self, session_id):     
        idx = self.find_session_on_buffer(session_id)
        
        if idx >= 0:
            return self.last_sessions_buffer[idx].item_ids
        else:
            return set()
   
    def get_sessions_with_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map[item_id]

    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors(self, session_items):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        candidate_neighbors_sessions = self.candidate_neighbor_sessions(session_items)
        candidate_neighbors_sessions_scores = self.calc_neighbor_sessions_scores(session_items, candidate_neighbors_sessions)
        
        candidate_neighbors_sessions_sorted = sorted(candidate_neighbors_sessions_scores, reverse=True, key=lambda x: x[1])
        candidate_neighbors_sessions_filtered = list(filter(lambda x: x[1] > 0.0 and x[1] < 1.0, candidate_neighbors_sessions_sorted))
        nearest_neighbors = candidate_neighbors_sessions_filtered[:self.nearest_neighbor_session_for_scoring]
        
        return nearest_neighbors       
        
    def candidate_neighbor_sessions(self, session_items):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling_strategy can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''

        # Retrieving all sessions from buffer which contains at least one item from the current sessions
        candidate_sessions_ids = list([session_id \
                                       for item_id in session_items \
                                       for session_id in self.get_sessions_with_item(item_id) \
                                       if self.find_session_on_buffer(session_id) != -1])

        if self.candidate_sessions_sample_size > 0 and len(candidate_sessions_ids) > self.candidate_sessions_sample_size:

            if self.sampling_strategy == 'recent':
                #As session_ids are the equivalent to timestamp of the first click in the session, ordering their ids is equivalent to sort by timestamp
                candidate_sessions_ids = sorted(candidate_sessions_ids, reverse=True)[:self.candidate_sessions_sample_size]  
            elif self.sampling_strategy == 'random':
                candidate_sessions_ids = random.sample(candidate_sessions_ids, self.candidate_sessions_sample_size)

        return candidate_sessions_ids   


    def score_items(self, neighbor_sessions_similarity):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        item_scores = defaultdict(int)
        # iterate over the sessions
        for session_id, similarity in neighbor_sessions_similarity:
            # get the items in this session                
            for item in self.get_session_items_from_buffer(session_id):
                item_scores[item] += similarity

        return item_scores 


    def calc_neighbor_sessions_scores(self, session_items, sessions_ids):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        

        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        for session in sessions_ids:
            cnt += 1
            # get items of the session, look up the cache first 
            neighbor_session_items = self.get_session_items_from_buffer(session)
            
            neighbor_session_score = self.score_neighbor_sessions(session_items, neighbor_session_items)
            if neighbor_session_score > 0:
                neighbors.append((session, neighbor_session_score))
                
        return neighbors

    def score_neighbor_sessions(self, session_items, neighbor_session_items):
        session_items_set = set(session_items)
        score = 0
        if self.first_session_clicks_decay == 'same':
            intersection = len(session_items_set & neighbor_session_items)
            if self.similarity == 'cosine':
                score = intersection / self.cosine_denominator(session_items_set, neighbor_session_items)
            elif self.similarity == 'jaccard':
                score = intersection / self.jaccard_denominator(session_items_set, neighbor_session_items)
            else:
                raise Exception("{} is not a valid similarity (cosine, jaccard)".format(self.similarity))
        else:
            decay_score = self.decay_score(session_items, neighbor_session_items)
            
            if self.similarity == 'cosine':
                score = decay_score / self.cosine_denominator(session_items_set, neighbor_session_items)
            elif self.similarity == 'jaccard':
                score = decay_score / self.jaccard_denominator(session_items_set, neighbor_session_items)
            else:
                raise Exception("{} is not a valid similarity (cosine, jaccard)".format(self.similarity))
        return score

    def decay_score(self, session_items, neighbor_session_items):
        score = 0
        #Loops over the reversed list of session items, to give more weight to last clicks
        for position, item in enumerate(reversed(session_items)):
            if item in neighbor_session_items:
                score += self.first_session_clicks_decay_fn(position+1)        
        return score
        

    def jaccard_denominator(self, first, second):
        union = len(first | second)
        return union    

    def cosine_denominator(self, first, second):
        return sqrt(len(first)) * sqrt(len(second))        

    def same_pos_decay(self, i):
        return 1

    def div_pos_decay(self, i):
        return 1/i

    def linear_pos_decay(self, i):
        return 1 - (0.1*i) if i <= 100 else 0    
    
    def log_pos_decay(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic_pos_decay(self, i):
        return 1/(i*i)
    