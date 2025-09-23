import math
import random
import numpy as np


class MetricScorer:

    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length > len(sorted_labels) or length <= 0:
            length = len(sorted_labels)
        return length

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer", ""), self.k)
        return self.__class__.__name__.replace("Scorer", "")

    def setLength(self, k):
        self.k = k


class APScorer(MetricScorer):

    def __init__(self, k=0):
        MetricScorer.__init__(self, k)

    def score(self, sorted_labels):
        nr_relevant = len([x for x in sorted_labels if x > 0])
        if nr_relevant == 0:
            return 0.0

        length = self.getLength(sorted_labels)
        ap = 0.0
        rel = 0

        for i in range(length):
            lab = sorted_labels[i]
            if lab >= 1:
                rel += 1
                ap += float(rel) / (i + 1.0)
        ap /= nr_relevant
        return ap



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mean_average_precision(queries, candidates, q_values, c_values):
    scorer = APScorer(candidates.shape[0])

    simmat = np.matmul(queries, candidates.T)

    ap_sum = 0
    for q in range(simmat.shape[0]):
        sim = simmat[q]
        index = np.argsort(sim)[::-1]
        sorted_labels = []
        for i in range(index.shape[0]):
            if c_values[index[i]] == q_values[q]:
                sorted_labels.append(1)
            else:
                sorted_labels.append(0)

        ap = scorer.score(sorted_labels)
        ap_sum += ap

    mAP = ap_sum / simmat.shape[0]

    return mAP

