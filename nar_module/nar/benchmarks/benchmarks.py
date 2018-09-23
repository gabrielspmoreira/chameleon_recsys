from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict

from ..evaluation import compute_metrics

class BenchmarkRecommender:
    def __init__(self, clicked_items_state, eval_benchmark_params, eval_streaming_metrics):
        self.clicked_items_state = clicked_items_state
        self.eval_benchmark_params = eval_benchmark_params
        self.streaming_metrics = eval_streaming_metrics
        
    def get_clf_suffix(self):
        return ''
        
    def get_description(self):
        return ''    
        
    def reset_eval_metrics(self):
        for metric in self.streaming_metrics:
            metric.reset()
        
    def train(self, users_ids, sessions_ids, sessions_items, sessions_next_items):
        pass
    
    def predict(self, users_ids, sessions_items, topk=5):
        pass
    
    def evaluate(self, users_ids, sessions_items, sessions_next_items, topk=5, eval_negative_items=None):
        sessions_next_items_expanded = np.expand_dims(sessions_next_items, axis=2)
        eval_negative_items_expanded = np.tile(np.expand_dims(eval_negative_items, axis=1), 
                                               reps=[1,sessions_next_items.shape[1],1])

        valid_items = np.concatenate([sessions_next_items_expanded, eval_negative_items_expanded], axis=2)

        
        preds = self.predict(users_ids, sessions_items, topk=topk, valid_items=valid_items)

        metrics_values = compute_metrics(preds, sessions_next_items, self.streaming_metrics, 
                                  metrics_suffix=self.get_clf_suffix())
        return metrics_values
    
    def _get_top_n_valid_items(self, items, topk, valid_items):
        count = 0        
        for item in items:
            if count == topk:
                break
            if (item in valid_items) or (valid_items is None):
                count += 1
                yield item
                
        #Returning expected top k predictions with 0 (padding item), to avoid errors when attributing to vectors with fixed size
        for i in range(count, topk):
            yield 0 


