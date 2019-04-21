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


def fig_prediction(pred,model_name,real):
    plt.figure(figsize=(10,5))
    for (c_pred,model_name) in zip(pred,model_name):
        plt.plot(c_pred,'.',label = model_name)
    plt.plot(real,'k.',label = 'real')
 
    plt.legend()
    plt.title('prediction')
    plt.savefig('./Figure/exp_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png')
    plt.show()

    print('regression results saved as following path :' + './Figure/exp_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return


def fig_metric(metric,model_name):
    assert(isinstance(metric,list) and isinstance(model_name,list) )

    plt.figure(figsize=(5,3))
    for (c_model_metric,c_model_name) in zip(metric,model_name):
        plt.plot(c_model_metric,label = c_model_name)

    plt.legend()
    plt.savefig('./Figure/metric_reg_' + time.ctime().replace(' ','_').replace(':','-')+'.png')
    plt.show()

    print('metric_reg saved as following path :' + './Figure/metric_reg_' + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return