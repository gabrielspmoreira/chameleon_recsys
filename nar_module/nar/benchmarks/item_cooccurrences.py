from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .benchmarks import BenchmarkRecommender

from ..utils import max_n_sparse_indexes


class ItemCooccurrenceRecommender(BenchmarkRecommender):
    
    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        #super(Instructor, self).__init__(name, year) #Python 2
        super().__init__(clicked_items_state, params, eval_streaming_metrics)
        
    def get_clf_suffix(self):
        return 'coocurrent'
        
    def get_description(self):
        return 'Most co-ocurrent in sessions'
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        pass
    
    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):
        session_predictions = np.zeros(dtype=np.int64,
                                         shape=[sessions_items.shape[0],
                                                sessions_items.shape[1],
                                                topk])
                     
        for row_idx, session_items in enumerate(sessions_items):            

            session_item_coocurrences = self.clicked_items_state.get_items_coocurrences()
            for col_idx, item in enumerate(session_items):
                if item != 0:
                    item_coocurrences = session_item_coocurrences[item]
                    sorted_items = max_n_sparse_indexes(item_coocurrences.data, item_coocurrences.indices, topn=len(item_coocurrences.data))
                    session_predictions[row_idx, col_idx] = list(self._get_top_n_valid_items(sorted_items.tolist(), topk, valid_items[row_idx, col_idx]))
            
        return session_predictions  