from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def compute_metrics(preds, labels, streaming_metrics, metrics_suffix=''):
    results = {}
    for metric in streaming_metrics:
        metric.add(preds, labels)
        result = metric.result()
        results['{}_{}'.format(metric.name(), metrics_suffix)] = result

    return results