'''
Malte Ludewig and Dietmar Jannach. 2018. Evaluation of Session-Based Recommenation
Algorithms. (2018). arXiv:1803.09587 [cs.IR] https://arxiv.org/abs/1803.09587
Inpired SR(sr.py), available in https://www.dropbox.com/sh/7qdquluflk032ot/AACoz2Go49q1mTpXYGe0gaANa?dl=0): ] 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter, defaultdict

from .benchmarks import BenchmarkRecommender

class SequentialRulesRecommender(BenchmarkRecommender):
    
    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        #super(Instructor, self).__init__(name, year) #Python 2
        super().__init__(clicked_items_state, params, eval_streaming_metrics)

        self.max_clicks_dist = params['max_clicks_dist'] #Max number of clicks to walk back in the session from the currently viewed item. (Default value: 10)        

        self.dist_between_clicks_decay = params['dist_between_clicks_decay'] #Decay function for distance between two items clicks within a session (linear, same, div, log, qudratic). (Default value: div)        
        self.dist_between_clicks_decay_fn = getattr(self, '{}_decay'.format(self.dist_between_clicks_decay))
        
        #Registering a state for this recommender which persists over TF Estimator train/eval loops
        if not self.get_clf_suffix() in clicked_items_state.benchmarks_states:
            clicked_items_state.benchmarks_states[self.get_clf_suffix()] = \
                                dict({'rules': defaultdict(dict) # Dict of Dict (Trained Rules)
                                     })
  
        state = clicked_items_state.benchmarks_states[self.get_clf_suffix()]
        self.rules = state['rules'] # Dict of Dict (Trained Rules)

    def get_clf_suffix(self):
        return 'sr'
        
    def get_description(self):
        return 'Sequential Rules'

    def get_all_sessions_clicks(self, sessions_items, sessions_next_items):
        sessions_all_items_but_last = list([list(filter(lambda x: x != 0, session)) for session in sessions_items])
        sessions_last_item_clicked = list([list(filter(lambda x: x != 0, session))[-1] for session in sessions_next_items])
        sessions_all_clicks = [previous_items + [last_item] \
                              for previous_items, last_item in zip(sessions_all_items_but_last, sessions_last_item_clicked)]
        return sessions_all_clicks
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        sessions_all_items = self.get_all_sessions_clicks(sessions_items, sessions_next_items)

        #For each session in the batch
        for session_items in sessions_all_items:
            #For each item of the session
            for i in range(1,len(session_items)):
                active_item = session_items[i]
                #For all items previously clicked in the session (limited by the max_clicks_dist)
                for j in range(max(0, i-self.max_clicks_dist),i):
                    past_item = session_items[j]
                    if not active_item in self.rules[past_item]:
                        self.rules[past_item][active_item] = 0.0                        
                    self.rules[past_item][active_item] += self.dist_between_clicks_decay_fn(i-j)


    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):
        session_predictions = np.zeros(dtype=np.int64,
                                         shape=[sessions_items.shape[0],
                                                sessions_items.shape[1],
                                                topk])
        
        for row_idx, session_items in enumerate(sessions_items):
            for col_idx, item in enumerate(session_items):
                if item != 0:
                    #Sorts items its score
                    preds = list(map(lambda y: y[0], sorted(self.rules[item].items(), reverse=True, key=lambda x: x[1])))
                    session_predictions[row_idx, col_idx] = list(self._get_top_n_valid_items(preds, topk, valid_items[row_idx, col_idx]))
                   
        return session_predictions


    def linear_decay(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same_decay(self, i):
        return 1
    
    def div_decay(self, i):
        return 1/i
    
    def log_decay(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic_decay(self, i):
        return 1/(i*i)