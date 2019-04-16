from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .benchmarks import BenchmarkRecommender

from ..utils import max_n_sparse_indexes

#Based on ItemKNN implentation of Balázs Hidasi (benchmark for GRU4Rec), also used and described in https://arxiv.org/abs/1803.09587
class ItemKNNRecommender(BenchmarkRecommender):
    
    def __init__(self, clicked_items_state, params, eval_streaming_metrics):
        #super(Instructor, self).__init__(name, year) #Python 2
        super().__init__(clicked_items_state, params, eval_streaming_metrics)

        #Regularization. Discounts the similarity of rare items (incidental co-occurrences). 
        self.reg_lambda = params['reg_lambda']
        #Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
        self.alpha = params['alpha']
        
    def get_clf_suffix(self):
        return 'item_knn'
        
    def get_description(self):
        return 'Item-KNN: Most similar items sessions based on normalized cosine similarity between session co-occurence'
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        pass
    
    def predict(self, users_ids, sessions_items, topk=5, valid_items=None):
        session_predictions = np.zeros(dtype=np.int64,
                                         shape=[sessions_items.shape[0],
                                                sessions_items.shape[1],
                                                topk])


        articles_support = self.clicked_items_state.get_articles_pop()
        articles_support_norm = np.power(articles_support + self.reg_lambda, self.alpha)
                     
        for row_idx, session_items in enumerate(sessions_items):            

            session_item_coocurrences = self.clicked_items_state.get_items_coocurrences()

            for col_idx, item in enumerate(session_items):
                if item != 0:
                    item_coocurrences = session_item_coocurrences[item]

                    norm = articles_support_norm * np.power(articles_support[item] + self.reg_lambda, 1.0-self.alpha)
                    #Hack to flatten the 1-row matrix to an array
                    items_similarity = np.array(item_coocurrences / norm)[0]
                    
                    sorted_items = max_n_sparse_indexes(items_similarity[item_coocurrences.indices],  #item_coocurrences.data, 
                            item_coocurrences.indices, topn=len(item_coocurrences.indices))
                    session_predictions[row_idx, col_idx] = list(self._get_top_n_valid_items(sorted_items.tolist(), topk, valid_items[row_idx, col_idx]))
            
        return session_predictions     