
import torch
import time
from data_manager import *
from SFTRL_CCFM import SFTRL_CCFM
from SFTRL_Vanila import SFTRL_Vanila
from peformance_manager import *




if __name__ == "__main__":

    nbUsers = 943
    nbMovies = 1682
    nbFeatures = nbUsers + nbMovies
    nbRatingsTrain = 90570
    nbRatingsTest = 9430

    data_dir = './Data/ml-100k/'
    filename1, filename2 = 'ub.base', './ub.test'


    # load dataset
    x_train, y_train, rate_train, timestamp_train = load_dataset(data_dir + filename1, nbRatingsTrain, nbFeatures, nbUsers)

    # sort dataset in time
    x_train_s, rate_train_s, _ = sort_dataset(x_train, rate_train, timestamp_train)

    #print(x_train[0:x_train_s.size:100])
    #print(rate_train_s[0:rate_train_s.size:100])

    down_sampling = 10

    # sparse to dense
    inputs_matrix = torch.tensor(x_train_s[0:x_train_s.size:down_sampling].todense()).double()
    outputs = torch.tensor(rate_train_s[0:x_train_s.size:down_sampling]).double()


    # model setup
    options = {}
    options['m']  = 10
    options['eta'] = 5e-2
    options['task'] = 'reg'

    options2 = {}
    options2['m']  = 10
    options2['eta'] = 5e-2
    options2['task'] = 'reg'


    # print(inputs_matrix)
    recent_num = -1
    Model_CCFM = SFTRL_CCFM(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options)
    Model_Vanila = SFTRL_Vanila(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options)


    pred_C , real = Model_CCFM.online_learning()
    pred_V, _ = Model_Vanila.online_learning()


    reg_metric_C = regression_metric(pred_C, real)
    reg_metric_V = regression_metric(pred_V, real)


    fig_prediction(pred_C, pred_V, real)

    fig_metric_reg(reg_metric_C, reg_metric_V)


    # fig = plt.figure()
    # plt.plot(pred_C,'b.',label = 'S_CCFM')
    # plt.plot(pred_V, 'g.',label = 'V_FM')
    # plt.plot(real,'r.',label = 'real')
    # plt.title('prediction')
    # plt.savefig('./Figure/exp_reg_' + time.ctime() +'.png')
    # plt.show()
    #
    #
    # fig = plt.figure()
    # plt.plot(reg_metric_C,'b',label = 'S_CCFM')
    # plt.plot(reg_metric_V,'g',label = 'V_FM')
    # plt.legend()
    # plt.savefig('./Figure/metric_reg_' + time.ctime() +'.png')
    # plt.show()


