from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .benchmarks import BenchmarkRecommender

class ContentBasedRecommender(BenchmarkRecommender):
    
    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        #super(Instructor, self).__init__(name, year) #Python 2
        super().__init__(clicked_items_state, params, eval_streaming_metrics)

    def get_clf_suffix(self):
        return 'cb'
        
    def get_description(self):
        return 'Content-Based similarity'
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        pass
    
    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):         
        acr_embeddings = self.eval_benchmark_params['content_article_embeddings_matrix']

        recent_items_buffer = self.clicked_items_state.get_recent_clicks_buffer()
        if valid_items is None:
            recent_unique_item_ids = np.unique([recent_items_buffer[np.nonzero(recent_items_buffer)]])            
        else:
            recent_unique_item_ids = np.unique(valid_items)
            
        acr_embeddings_recent_items = acr_embeddings[recent_unique_item_ids]


        session_predictions = np.zeros(dtype=np.int64,
                                       shape=[sessions_items.shape[0],
                                              sessions_items.shape[1],
                                              topk])

        for row_idx, session_items in enumerate(sessions_items):    

            for col_idx, item in enumerate(session_items):
                if item != 0:

                    #Computing cosine similarity between this item and all recent items (from buffer)
                    #P.s. Do not need to ignore the current item (whose similarity is always, because this item will not be among the valid items (next click + negative samples not present in the session))
                    similarities = cosine_similarity(acr_embeddings[item].reshape(1, -1), 
                                                     acr_embeddings_recent_items)[0]
                    similar_items_sorted_idx = np.argsort(similarities, axis=0)[::-1]
                    similar_items_ids = recent_unique_item_ids[similar_items_sorted_idx]

                    session_predictions[row_idx, col_idx] = list(self._get_top_n_valid_items(similar_items_ids, topk, valid_items[row_idx, col_idx]))
                    

        return session_predictions