import numpy as np
import scipy.io as sio
from scipy.sparse import lil_matrix
from scipy.sparse import hstack

from datetime import timezone, timedelta, datetime
import csv


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt



def load_dataset(filename, lines, columns , nbUsers):
    # Features are one-hot encoded in a sparse matrix
    X = lil_matrix((lines, columns)).astype('float32')
    X2 = lil_matrix((lines, columns +1 )).astype('float64')
    # Labels are stored in a vector
    Y = []
    Y2 = []
    date_time = []
    line = 0
    with open(filename, 'r') as f:
        samples = csv.reader(f, delimiter='\t')
        for userId, movieId, rating, timestamp in samples:
            X[line, int(userId) - 1] = 1
            X[line, int(nbUsers) + int(movieId) - 1] = 1

            X2[line, int(userId) - 1] = 1
            X2[line, int(nbUsers) + int(movieId) - 1] = 1
            X2[line, columns ] = 1

            if int(rating) >= 3:
                Y.append(1)
            else:
                Y.append(0)
            line = line + 1
            date_time.append(timestamp)
            Y2.append(rating)

    date_time = np.array(date_time).astype('float32')
    Y = np.array(Y).astype('float64')
    return X,X2 , Y, Y2, date_time



def sort_dataset(X,Y,utc_time_stamp):

    # make up some data
    time_order = np.array([datetime.utcfromtimestamp(current_utc) for current_utc in utc_time_stamp])

    #x = np.array([datetime.utcfromtimestamp(current_utc) for current_utc in timestamp_train])
    Y = np.array([float(i) for i in Y])

    sorted_X = X[time_order.argsort()]
    sorted_Y = Y[time_order.argsort()]

    return sorted_X,sorted_Y,time_order









#if __name__ == "__main__" :

    # nbUsers = 943
    # nbMovies = 1682
    # nbFeatures = nbUsers + nbMovies
    # nbRatingsTrain = 90570
    # nbRatingsTest = 9430
    #
    # data_dir = './Data/ml-100k/'
    # filename1, filename2 = 'ub.base', './ub.test'
    #
    #
    # # load dataset
    # x_train, y_train, rate_train, timestamp_train = load_dataset(data_dir + filename1, nbRatingsTrain, nbFeatures, nbUsers)
    #
    # # sort dataset in time
    # x_train_s, rate_train_s, _ = sort_dataset(x_train, rate_train, timestamp_train)
    #
    #
    # # sparse to dense
    # inputs_matrix = torch.tensor(x_train_s.todense()).double()
    # outputs = torch.tensor(rate_train_s).double()
    #
    #
    #
    # # model setup
    # options = {}
    # options['m']  = 20
    # options['eta'] = 5e-2
    # options['task'] = 'reg'
    #
    # #print(inputs_matrix)
    #
    #
    # recent_num = 1000
    #
    # Model = SFTRL(inputs_matrix[:recent_num,:],outputs[:recent_num],options)
    # #alpha = torch.tensor(np.random.randn(10,1))
    # pred,real = Model.online_learning()

    # plt.figure()
    # plt.plot(pred,'b.')
    # plt.plot(real,'r.')
    # plt.savefig('./demo.png')
    # plt.show()
