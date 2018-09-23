from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter

from .benchmarks import BenchmarkRecommender

class RecentlyPopularRecommender(BenchmarkRecommender):
    
    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        #super(Instructor, self).__init__(name, year) #Python 2
        super().__init__(clicked_items_state, params, eval_streaming_metrics)
        
    def get_clf_suffix(self):
        return 'pop_recent'
        
    def get_description(self):
        return 'Most Popular from Recently Clicked'
    
    def get_recent_popular_item_ids(self):
        recent_items_buffer = self.clicked_items_state.get_recent_clicks_buffer()
        recent_items_buffer_nonzero = recent_items_buffer[np.nonzero(recent_items_buffer)]
        #Dealing with first batch, when there is no item in the buffer yet
        if len(recent_items_buffer_nonzero) == 0:
            recent_items_buffer_nonzero = [0]
        item_counter = Counter(recent_items_buffer_nonzero)
        popular_item_ids, popular_items_count = zip(*item_counter.most_common())
        return popular_item_ids, popular_items_count
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        pass
    
    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):
        popular_item_ids, popular_items_count = self.get_recent_popular_item_ids()
        
        session_predictions = np.zeros(dtype=np.int64,
                                         shape=[sessions_items.shape[0],
                                                sessions_items.shape[1],
                                                topk])
        
        for row_idx, session_items in enumerate(sessions_items):
            for col_idx, item in enumerate(session_items):
                if item != 0:
                    session_predictions[row_idx, col_idx] = list(self._get_top_n_valid_items(popular_item_ids, topk, valid_items[row_idx, col_idx]))
                   
        return session_predictions