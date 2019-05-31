import numpy as np
import torch
import time
import matplotlib.pyplot as plt


def regression_metric(pred,real):
    val = 0.0
    metric = [np.inf]
    for idx,(pred_idx,real_idx) in enumerate(zip(pred,real)):
        val += (pred_idx - real_idx)**2
        metric.append( (1/(idx + 1) )* val)
    metric = np.asarray(metric).reshape([-1,1])

    return metric


def classfication_metric(pred,real):
    val = 0.0
    metric = [np.inf]
    metric_cnt = [np.inf]
    for idx,(pred_idx,real_idx) in enumerate(zip(pred,real)):
        val += 1 if pred_idx == real_idx else 0
        metric.append( 1/(idx + 1)* np.log( 1.0 + np.exp(-pred_idx*real_idx) )    )
        metric_cnt.append( 1/(idx + 1)*val )
    metric = np.asarray(metric).reshape([-1,1])
    metric_cnt = np.asarray(metric_cnt).reshape([-1, 1])

    return metric,metric_cnt



# def fig_prediction(pred, model_name ,real, option, save_path):
#     plt.figure(figsize=(10,5))
#     for (c_pred,model_name) in zip(pred,model_name):
#         plt.plot(c_pred,'.',label = model_name ,alpha = 0.4)
#     plt.plot(real,'k.',label = 'real' ,alpha = 0.4)
#
#     plt.xlabel('iteration')
#     plt.ylabel('rating')
#
#     #plt.ylim([1900,2100])
#     plt.legend()
#     plt.title(option['Data'] + ' prediction' + ' eta_' + str(option['eta']))
#     plt.savefig(save_path + time.ctime().replace(' ','_').replace(':','-') +'.png')
#     plt.show()
#
#     print('regression results saved as following path :' + save_path + time.ctime().replace(' ','_').replace(':','-') +'.png ')
#     return

def fig_prediction(pred, model_name ,real, option, save_path):
    fig = plt.figure(figsize=(10,15))
    for ith,(c_pred,model_name_ith) in enumerate(zip(pred,model_name)):
        fig.add_subplot(len(model_name),1,ith + 1)
        plt.plot(c_pred,'.',label = model_name_ith ,alpha = 0.4)
        plt.plot(real,'k.',label = 'real' ,alpha = 0.4)
        plt.xlabel('iteration')
        plt.ylabel('rating')

        #plt.ylim([1900,2100])
        plt.legend()
        plt.title(option['Data'] + ' prediction' + ' eta_' + str(option['eta']))
    plt.savefig(save_path + time.ctime().replace(' ','_').replace(':','-') +'.png')
    plt.show()

    print('regression results saved as following path :' + save_path + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return



def fig_metric_reg(metric,model_name,option,save_path):
    assert(isinstance(metric,list) and isinstance(model_name,list) )

    plt.figure(figsize=(10,5))
    for (c_model_metric,c_model_name) in zip(metric,model_name):
        plt.plot(c_model_metric,label = c_model_name ,alpha = 0.9)

    plt.legend()


    plt.xlabel('iteration')
    plt.ylabel('mean of accumulated loss')
    plt.title(option['Data'] + ' metric' + ' eta_' + str(option['eta']))
    plt.savefig(save_path + time.ctime().replace(' ','_').replace(':','-')+'.png')
    plt.show()
    print('metric saved as following path :' + save_path + time.ctime().replace(' ','_').replace(':','-') +'.png ')
    return


def fig_metric_cls(metric, metric_cnt, model_name, option, save_path):
    assert (isinstance(metric, list) and isinstance(model_name, list))

    plt.figure(figsize=(10, 5))
    for (c_model_metric, c_model_name) in zip(metric, model_name):
        plt.plot(c_model_metric, label=c_model_name, alpha=0.9)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('mean of accumulated loss')
    plt.title(option['Data'] + ' metric' + ' eta_' + str(option['eta']))
    plt.savefig(save_path + time.ctime().replace(' ', '_').replace(':', '-') + '.png')
    plt.show()
    print('metric cls saved as following path :' + save_path + time.ctime().replace(' ', '_').replace(':', '-') + '.png ')

    plt.figure(figsize=(10, 5))
    for (c_model_metric, c_model_name) in zip(metric_cnt, model_name):
        plt.plot(c_model_metric, label=c_model_name, alpha=0.9)

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('mean of accumulated loss')
    plt.title(option['Data'] + ' metric' + ' eta_' + str(option['eta']))
    plt.savefig(save_path + 'cnt_'+ time.ctime().replace(' ', '_').replace(':', '-')  + '.png')
    plt.show()
    print('metric cls cnt saved as following path :' + save_path + time.ctime().replace(' ', '_').replace(':', '-') + '.png ')

    return
