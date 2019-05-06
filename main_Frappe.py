from data_manager import *
from models.SFTRL_CCFM import SFTRL_CCFM
from models.SFTRL_Vanila import SFTRL_Vanila
from models.FM_FTRL import  FM_FTRL
from models.RRF import RRF

from peformance_manager import *


if __name__ == "__main__":

    # filename = './data/frappe/'
    # filename1 = 'frappe'
    # x_train_s, rate_train_s = load_dataset_YearPredictionMSD(filename)

    filename = './data/frappe/frappe.train.libfm'
    filename1 = 'frappe_train'
    x_train_s, rate_train_s = load_dataset_fappe(filename, logloss_opt=False)


    want_permute = True
    if want_permute :
        idx = np.random.permutation(x_train_s.shape[0])
        x_train_s = x_train_s[idx]
        rate_train_s = rate_train_s[idx]
    else:
        pass

    down_sampling = 50
    if down_sampling > 0 :
        # x_train_s2 = x_train_s[0:x_train_s.size:down_sampling]
        # rate_train_s2 = rate_train_s[0:x_train_s.size:down_sampling]
        x_train_s2 = x_train_s[1:x_train_s.size:down_sampling]
        rate_train_s2 = rate_train_s[1:x_train_s.size:down_sampling]
    else:
        pass
    inputs_matrix = torch.tensor(x_train_s2).double()
    outputs = torch.tensor(rate_train_s2).double()


    c = RRF(loss='logit', task="classification", learning_rate=0.003,
            learning_rate_gamma=0.001, gamma = .1, D = 50)


    pred_RRF = c.fit(x_train_s[0:x_train_s.size:down_sampling], rate_train_s[0:x_train_s.size:down_sampling])
    #train_time = c.train_time
    #print(pred_rrf)

    m = 20

    options = {}
    options['Data'] = filename
    options['m']  = m
    options['eta'] = 5e-2
    options['task'] = 'cls'

    options2 = {}
    options2['Data'] = filename
    options2['m']  = m
    options2['eta'] = 5e-2
    options2['task'] = 'cls'

    options3 = {}
    options3['Data'] = filename
    options3['m']  = m
    options3['eta'] = 5e-2
    options3['task'] = 'cls'
    #
    #
    recent_num = -1
    Model_CCFM = SFTRL_CCFM(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options)
    Model_Vanila = SFTRL_Vanila(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options2)
    Model_FM_FTRL = FM_FTRL(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options3)


    pred_C , real = Model_CCFM.online_learning()
    pred_V, _ = Model_Vanila.online_learning()
    pred_F, _ = Model_FM_FTRL.online_learning()


    cls_metric_C,cls_metric_C_cnt = classfication_metric(pred_C, real)
    cls_metric_V,cls_metric_V_cnt = classfication_metric(pred_V, real)
    cls_metric_F,cls_metric_F_cnt = classfication_metric(pred_F, real)
    cls_metric_RRF,cls_metric_RRF_cnt = classfication_metric(pred_RRF, real)


    save_path1 = './figure_results/frappe/pred_cls_'
    save_path2 = './figure_results/frappe/metric_cls_'


    save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF','FM_FTRL']
    fig_prediction([pred_C,pred_V,pred_RRF,pred_F],save_legend,real,options,save_path1)
    fig_metric_cls([cls_metric_C,cls_metric_V,cls_metric_RRF,cls_metric_F],
                   [cls_metric_C_cnt,cls_metric_V_cnt,cls_metric_RRF_cnt,cls_metric_F_cnt],save_legend,options,save_path2)
