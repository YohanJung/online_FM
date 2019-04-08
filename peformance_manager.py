import numpy as np


def regression_metric(pred,real):


    val = 0.0
    metric = []
    for idx,(pred_idx,real_idx) in range(enumerate(zip(pred,real))):

        val += (pred_idx - real_idx)**2
        metric.append(1/(idx+1)*val)

    return np.asarray(metric)

def classfication_metric():

    return