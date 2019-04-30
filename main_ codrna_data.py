from data_manager import *
from models.SFTRL_CCFM import SFTRL_CCFM
from models.SFTRL_Vanila import SFTRL_Vanila
from models.FM_FTRL import  FM_FTRL
from models.RRF import RRF
from peformance_manager import *
from sklearn.datasets import load_svmlight_file

#import

if __name__ == "__main__":

    data_dir = './data/cod-rna2/'
    filename = "cod-rna2.scale"
    X, y = load_svmlight_file(data_dir + filename, n_features=8)
    X = X.toarray()

    want_permute = False
    if want_permute :
        idx = np.random.permutation(X.shape[0])
        x_train_s = np.asarray(X[idx])
        rate_train_s = np.asarray(y[idx])
    else:
        x_train_s = np.asarray(X)
        rate_train_s = np.asarray(y)


    down_sampling = 5
    # sparse to dense
    inputs_matrix = torch.tensor(x_train_s[0:x_train_s.size:down_sampling]).double()
    outputs = torch.tensor(rate_train_s[0:x_train_s.size:down_sampling]).double()

    # # model setup

    c = RRF(loss='l2', task="regression", learning_rate=0.003,
            learning_rate_gamma=0.001, gamma=1.0, D=100)


    pred_RRF = c.fit(x_train_s[0:x_train_s.size:down_sampling], rate_train_s[0:x_train_s.size:down_sampling])
    #train_time = c.train_time
    #print(pred_rrf)

    m = 10

    options = {}
    options['Data'] = filename
    options['m']  = m
    options['eta'] = 5e-2
    options['task'] = 'reg'

    options2 = {}
    options2['Data'] = filename
    options2['m']  = m
    options2['eta'] = 5e-2
    options2['task'] = 'reg'

    options3 = {}
    options3['Data'] = filename
    options3['m']  = m
    options3['eta'] = 5e-2
    options3['task'] = 'reg'
    #
    #
    recent_num = -1
    Model_CCFM = SFTRL_CCFM(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options)
    Model_Vanila = SFTRL_Vanila(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options2)
    Model_FM_FTRL = FM_FTRL(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options3)


    pred_C , real = Model_CCFM.online_learning()
    pred_V, _ = Model_Vanila.online_learning()
    pred_F, _ = Model_FM_FTRL.online_learning()


    reg_metric_C = regression_metric(pred_C, real)
    reg_metric_V = regression_metric(pred_V, real)
    reg_metric_F = regression_metric(pred_F, real)
    reg_metric_RRF = regression_metric(pred_RRF, real)

    #print(reg_metric_C)

    # save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF']
    # fig_prediction2([pred_C,pred_V,pred_RRF], save_legend,real,options)
    # fig_metric2([reg_metric_C,reg_metric_V,reg_metric_RRF], save_legend,options)


    save_path1 = './figure_results/cod-rna2/pred_reg_'
    save_path2 = './figure_results/cod-rna2/metric_reg_'

    # save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF']
    # fig_prediction2([pred_C,pred_V,pred_RRF],save_legend,real,options,save_path1)
    # fig_metric2([reg_metric_C,reg_metric_V,reg_metric_RRF], save_legend,options,save_path2)


    save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF','FM_FTRL']
    fig_prediction2([pred_C,pred_V,pred_RRF,pred_F],save_legend,real,options,save_path1)
    fig_metric2([reg_metric_C,reg_metric_V,reg_metric_RRF,reg_metric_F], save_legend,options,save_path2)

    # # save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','FM_FTRL']
    # # fig_prediction2([pred_C,pred_V,pred_F], save_legend,real,options)
    # # fig_metric2([reg_metric_C,reg_metric_V,reg_metric_F], save_legend,options)
