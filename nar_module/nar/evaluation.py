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