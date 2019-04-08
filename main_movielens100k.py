
from data_manager import *
from SFTRL import SFTRL
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


    #print(x_train)

    # sparse to dense
    inputs_matrix = torch.tensor(x_train_s.todense()).double()
    outputs = torch.tensor(rate_train_s).double()



    # model setup
    options = {}
    options['m']  = 10
    options['eta'] = 5e-2
    options['task'] = 'reg'

    # print(inputs_matrix)


    recent_num = 1000

    Model = SFTRL(inputs_matrix[:recent_num ,:] ,outputs[:recent_num] ,options)
    pred ,real = Model.online_learning()

    fig = plt.figure()
    plt.plot(pred,'b.')
    plt.plot(real,'r.')
    plt.savefig('./Figure/demo.png')
    plt.show()

    #reg_metric = regression_metric(pred, real)


    # fig = plt.figure()
    # fig.add_subplot(1,2,1)
    # plt.plot(pred,'b.')
    # plt.plot(real,'r.')
    #
    # fig.add_subplot(1,2,2)
    # plt.plot(reg_metric)
    # plt.savefig('./Figure/demo.png')
    # plt.show()