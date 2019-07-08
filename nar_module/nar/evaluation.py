from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from time import time

from .metrics import HitRate, HitRateBySessionPosition, MRR, ItemCoverage, PopularityBias, CategoryExpectedIntraListDiversity, Novelty, ExpectedRankSensitiveNovelty, ExpectedRankRelevanceSensitiveNovelty, ContentExpectedRankSensitiveIntraListDiversity, ContentExpectedRankRelativeSensitiveIntraListDiversity, ContentExpectedRankRelevanceSensitiveIntraListDiversity, ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity, ContentAverageIntraListDiversity, ContentMedianIntraListDiversity, ContentMinIntraListDiversity


def update_metrics(preds, labels, labels_norm_pop, preds_norm_pop, clicked_items, streaming_metrics, recommender=''):
    for metric in streaming_metrics:
        if metric.name == HitRateBySessionPosition.name:
            #if recommender == 'chameleon':
                metric.add(preds, labels, labels_norm_pop)
        else:
            if metric.name == ItemCoverage.name:
                metric.add(preds, labels, clicked_items)
            elif metric.name in [PopularityBias.name,
                                 Novelty.name,
                                 ExpectedRankSensitiveNovelty.name,
                                 ExpectedRankRelevanceSensitiveNovelty.name]:
                metric.add(preds, labels, preds_norm_pop)
            else:
                metric.add(preds, labels)

def compute_metrics_results(streaming_metrics, recommender=''):
    results = {}
    for metric in streaming_metrics:
        if metric.name == HitRateBySessionPosition.name:
            #if recommender == 'chameleon':
                recall_by_session_pos, avg_norm_pop_by_session_pos, hitrate_total_by_session_pos = metric.result()
                for key in recall_by_session_pos:
                    results['{}_{}_{:02d}'.format(metric.name, recommender, key)] = recall_by_session_pos[key]
                    if recommender == 'chameleon':
                        results['{}_{}_{:02d}'.format('clicks_at_pos', recommender, key)] = hitrate_total_by_session_pos[key]
                        results['{}_{}_{:02d}'.format('avg_norm_pop_by_pos', recommender, key)] = avg_norm_pop_by_session_pos[key]
        else:
            result = metric.result()
            results['{}_{}'.format(metric.name, recommender)] = result

    return results

class ColdStartAnalysisState():
 
    def __init__(self):
        self.items_num_steps_before_first_rec = dict()
 
    def update_items_num_steps_before_first_rec(self, batch_rec_items, items_first_click_step, step):
        batch_top_rec_ids_flatten = batch_rec_items.reshape(-1)
        batch_top_rec_ids_nonzero = batch_top_rec_ids_flatten[np.nonzero(batch_top_rec_ids_flatten)]
        batch_top_rec_ids_set = set(batch_top_rec_ids_nonzero)
 
        for item_id in batch_top_rec_ids_set:
            if item_id in items_first_click_step and \
               item_id not in self.items_num_steps_before_first_rec:
                elapsed_steps = step - items_first_click_step[item_id]
                assert elapsed_steps >= 0
                self.items_num_steps_before_first_rec[item_id] = elapsed_steps
 
 
    def get_statistics(self):
        if len(self.items_num_steps_before_first_rec) > 0:
            values = np.array(list(self.items_num_steps_before_first_rec.values()))
            stats = {'min': np.min(values),
                    '01%': np.percentile(values, 1),
                    '10%': np.percentile(values, 10),
                    '25%': np.percentile(values, 25),
                    '50%': np.percentile(values, 50),
                    '75%': np.percentile(values, 75),
                    '90%': np.percentile(values, 90),
                    '99%': np.percentile(values, 99),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                     }
        else:
            stats = {'count': 0}
 
        return stats