import numpy as np
import torch
import time
import matplotlib.pyplot as plt


def regression_metric(pred,real):
    val = 0.0
    metric = [np.inf]
    for idx,(pred_idx,real_idx) in enumerate(zip(pred,real)):
        val += (pred_idx - real_idx)**2
        metric.append(1 / (idx + 1) * val)
    metric = np.asarray(metric).reshape([-1,1])

    return metric


def classfication_metric():
    return


def fig_prediction(pred,model_name,real,option):
    plt.figure(figsize=(10,5))
    for (c_pred,model_name) in zip(pred,model_name):
        plt.plot(c_pred,'.',label = model_name ,alpha = 0.5)
    plt.plot(real,'k.',label = 'real')

    plt.xlabel('iteration')
    plt.ylabel('rating')
    plt.ylim([0.0 - 0.1 , 6.0 + .1])
    plt.legend()
    plt.title(option['Data'] + ' prediction' + ' eta_' + str(option['eta']))
    plt.savefig('./Figure/exp_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png')
    plt.show()

    print('regression results saved as following path :' + './Figure/exp_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return


def fig_metric(metric,model_name,option):
    assert(isinstance(metric,list) and isinstance(model_name,list) )

    plt.figure(figsize=(10,5))
    for (c_model_metric,c_model_name) in zip(metric,model_name):
        plt.plot(c_model_metric,label = c_model_name ,alpha = 0.9)

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('mean of accumulated loss')
    plt.title(option['Data'] + ' metric' + ' eta_' + str(option['eta']))
    plt.savefig('./Figure/metric_reg_' + time.ctime().replace(' ','_').replace(':','-')+'.png')
    plt.show()
    print('metric_reg saved as following path :' + './Figure/metric_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return