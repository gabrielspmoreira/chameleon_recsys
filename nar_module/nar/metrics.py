from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import defaultdict


class StreamingMetric:
    def __init__(self, topn):
        self.topn = topn
        self.reset()

    def name(self):
        pass

    def reset(self):
        pass

    def add(self, predictions, labels):
        pass

    def result(self):
        pass


class MRR(StreamingMetric):

    def __init__(self, topn):
        super().__init__(topn)
        self.reset()

    def name(self):
        return 'mrr_at_n'

    def reset(self):
        self.mrr_results = []

    def add(self, predictions, labels):
        measures = []
        for row_idx, session_labels in enumerate(labels):
                for col_idx, item_label in enumerate(session_labels):
                    if item_label != 0:
                        correct_preds = (item_label == predictions[row_idx, col_idx])[:self.topn].astype(np.int32)
                        correct_preds_pos = np.where(correct_preds)[0]

                        reciprocal_rank = 0
                        if len(correct_preds_pos) > 0:
                            reciprocal_rank = 1.0 / (1+correct_preds_pos[0])
                        measures.append(reciprocal_rank)       
        self.mrr_results.extend(measures)

    def result(self):
        avg_mrr = np.mean(self.mrr_results)
        return avg_mrr


class NDCG(StreamingMetric):

    def __init__(self, topn):
        super().__init__(topn)
        self.reset()

    def name(self):
        return 'ndcg_at_n'

    def reset(self):
        self.ndcg_results = []

    def add(self, predictions, labels):
        measures = []
        for row_idx, session_labels in enumerate(labels):
                for col_idx, item_label in enumerate(session_labels):
                    if item_label != 0:
                        correct_preds = (item_label == predictions[row_idx, col_idx]).astype(np.int32)
                        ndcg = StreamingMetrics._ndcg_at_k(correct_preds, self.topn)
                        measures.append(ndcg)                        
        self.ndcg_results.extend(measures)

    def result(self):
        avg_ndcg = np.mean(self.ndcg_results)
        return avg_ndcg

    @staticmethod
    def _ndcg_at_k(r, k):
        #Based on https://gist.github.com/bwhite/3726239, but with alternative formulation of DCG
        #which places stronger emphasis on retrieving relevant documents (used in Kaggle)
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return np.sum((np.power(2,r)-1) / np.log2(np.arange(2, r.size + 2)))
            return 0.
    
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return dcg_at_k(r, k) / dcg_max 


class HitRate(StreamingMetric):

    def __init__(self, topn):
        super().__init__(topn)
        self.reset()

    def name(self):
        return 'hitrate_at_n'

    def reset(self):
        self.hitrate_total = 0
        self.hitrate_matches = 0 

    def add(self, predictions, labels):
        total = 0
        matches = 0
        for row_idx, session_labels in enumerate(labels):
                for col_idx, item_label in enumerate(session_labels):
                    if item_label != 0:
                        total += 1
                        if item_label in predictions[row_idx, col_idx][:self.topn]:
                            matches += 1
        self.hitrate_total += total
        self.hitrate_matches += matches

    def result(self):
        hitrate = self.hitrate_matches / float(self.hitrate_total)
        return hitrate

class HitRateBySessionPosition(StreamingMetric):

    def __init__(self, topn):
        super().__init__(topn)
        self.reset()

    def name(self):
        return 'hitrate_by_pos_at_n'

    def reset(self):
        self.hitrate_matches_by_session_pos = defaultdict(int)
        self.hitrate_total_by_session_pos = defaultdict(int)

    def add(self, predictions, labels):
        for row_idx, session_labels in enumerate(labels):
                for col_idx, item_label in enumerate(session_labels):
                    if item_label != 0:
                        self.hitrate_total_by_session_pos[col_idx+1] += 1                        
                        if item_label in predictions[row_idx, col_idx][:self.topn]:
                            self.hitrate_matches_by_session_pos[col_idx+1] += 1       

    def result(self):
        hitrate_by_session_pos = dict([(key, (self.hitrate_matches_by_session_pos[key] if key in self.hitrate_matches_by_session_pos else 0) \
                                            / float(self.hitrate_total_by_session_pos[key])) \
                                      for key in self.hitrate_total_by_session_pos])
        return hitrate_by_session_pos#, self.hitrate_total_by_session_pos 
