from data_manager import *
from models.SFTRL_CCFM import SFTRL_CCFM
from models.SFTRL_Vanila import SFTRL_Vanila
from models.FM_FTRL import  FM_FTRL
from models.RRF import RRF

from peformance_manager import *


if __name__ == "__main__":

    nbUsers = 943
    nbMovies = 1682
    nbFeatures = nbUsers + nbMovies
    nbRatingsTrain = 90570
    nbRatingsTest = 9430

    data_dir = './data/ml-100k/'
    #filename1, filename2 = 'ub.base', './ub.test'
    filename1, filename2 = 'ua.base', './ua.test'

    # load dataset
    _, x_train, y_train, rate_train, timestamp_train = load_dataset_movielens(data_dir + filename1, nbRatingsTrain, nbFeatures, nbUsers)
    # sort dataset in time
    #x_train_s, rate_train_s, _ = sort_dataset(x_train, rate_train, timestamp_train)
    x_train_s, rate_train_s, _ = sort_dataset_movielens(x_train, rate_train, timestamp_train)

    want_permute = True
    if want_permute :
        idx = np.random.permutation(x_train_s.shape[0])
        x_train_s = x_train_s[idx]
        rate_train_s = rate_train_s[idx]
    else:
        pass


    down_sampling = 50
    # sparse to dense

    # # model setup


    c = RRF(loss='l2', task="regression", learning_rate=0.01,
            learning_rate_gamma=0.001, gamma = 1.0, D = 50)
    pred_RRF = c.fit(x_train_s.todense()[0:x_train_s.todense().size:down_sampling], rate_train_s[0:x_train_s.todense().size:down_sampling])



    # sparse to dense
    inputs_matrix = torch.tensor(x_train_s[0:x_train_s.size:down_sampling].todense()).double()
    outputs = torch.tensor(rate_train_s[0:x_train_s.size:down_sampling]).double()

    # # model setup

    m = 10

    options = {}
    options['Data'] = filename1
    options['m']  = m
    options['eta'] = 5e-2
    options['task'] = 'reg'

    options2 = {}
    options2['Data'] = filename1
    options2['m']  = m
    options2['eta'] = 5e-2
    options2['task'] = 'reg'

    options3 = {}
    options3['Data'] = filename1
    options3['m']  = m
    options3['eta'] = 5e-2
    options3['task'] = 'reg'
    #
    #
    recent_num = -1
    Model_CCFM = SFTRL_CCFM(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options)
    Model_Vanila = SFTRL_Vanila(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options2)
    Model_FM_FTRL = FM_FTRL(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options3)


    print(Model_CCFM.num_feature)

    pred_C , real = Model_CCFM.online_learning()
    pred_V, _ = Model_Vanila.online_learning()
    pred_F, _ = Model_FM_FTRL.online_learning()


    reg_metric_C = regression_metric(pred_C, real)
    reg_metric_V = regression_metric(pred_V, real)
    reg_metric_F = regression_metric(pred_F, real)
    reg_metric_RRF = regression_metric(pred_RRF, real)

    # save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila']
    # fig_prediction([pred_C,pred_V], save_legend,real,options)
    # fig_metric([reg_metric_C,reg_metric_V], save_legend,options)

    #print(reg_metric_C)

    # save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF']
    # fig_prediction([pred_C,pred_V,pred_RRF], save_legend,real,options)
    # fig_metric([reg_metric_C,reg_metric_V,reg_metric_RRF], save_legend,options)

    save_path1 = './figure_results/movielens/pred_reg_'
    save_path2 = './figure_results/movielens/metric_reg_'
    save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF']
    fig_prediction([pred_C,pred_V,pred_RRF],save_legend,real,options,save_path1)
    fig_metric([reg_metric_C,reg_metric_V,reg_metric_RRF], save_legend,options,save_path2)

    # save_legend = ['SFTRL_CCFM', 'SFTRL_Vanila','RRF','FM_FTRL']
    # fig_prediction([pred_C,pred_V,pred_RRF,pred_F],save_legend,real,options,save_path)
    # fig_metric([reg_metric_C,reg_metric_V,reg_metric_RRF,reg_metric_F], save_legend,options,save_path)