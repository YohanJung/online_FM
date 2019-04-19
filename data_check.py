from data_manager import *
from peformance_manager import *

#import matplotlib
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt



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

    down_sampling = 10
    # sparse to dense
    inputs_matrix = torch.tensor(x_train_s[0:x_train_s.size:down_sampling].todense()).double()
    outputs = torch.tensor(rate_train_s[0:x_train_s.size:down_sampling]).double()

    #plt.interactive(b = True)
    plt.figure()
    for i_th in inputs_matrix:
        plt.plot( np.arange(i_th.numpy().size),i_th.numpy(),'.')
        #print(i_th.numpy())
    plt.savefig('./Figure/data_check_' + time.ctime() +'.png')
    plt.show()

    print('metric_reg saved as following path :' + './Figure/data_check_' + time.ctime() +'.png ')
    #print(inputs_matrix.sum(1))