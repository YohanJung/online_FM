import numpy as np
import torch
import time
import matplotlib.pyplot as plt


def regression_metric(pred,real):
    val = torch.tensor([0]).double()
    metric = []
    for idx,(pred_idx,real_idx) in enumerate(zip(pred,real)):
        val += (pred_idx - real_idx)**2
        metric.append(1 / (idx + 1) * val)
    metric = np.asarray(metric).reshape([-1,1])

    return metric


def classfication_metric():
    return


def fig_prediction(pred_C,pred_V,real):
    fig = plt.figure(figsize=(5,3))
    plt.plot(pred_C.data,'b.',label = 'S_CCFM')
    plt.plot(pred_V.data, 'g.',label = 'V_FM')
    plt.plot(real,'r.',label = 'real')
    plt.title('prediction')
    plt.savefig('./Figure/exp_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png')
    plt.show()

    print('regression results saved as following path :' + './Figure/exp_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return

def fig_metric_reg(reg_metric_C,reg_metric_V):
    fig = plt.figure(figsize=(5,3))
    plt.plot(reg_metric_C,'b',label = 'S_CCFM')
    plt.plot(reg_metric_V,'g',label = 'V_FM')
    plt.legend()
    plt.savefig('./Figure/metric_reg_' + time.ctime().replace(' ','_').replace(':','-')+'.png')
    plt.show()

    print('metric_reg saved as following path :' + './Figure/metric_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png ')


    return