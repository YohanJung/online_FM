import numpy
import torch
from torch.autograd import Variable
from torch.nn import Module

import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

#tensor_type = torch.DoubleTensor
numpy_type = np.float64


# Ad Click Prediction: a View from the Trenches , H. Brendan McMahan el al.
# Factorization Machines with Follow-The-Regularized-Leader for CTR prediction in Display Advertising , Anh-Phuong TA


def load_data(filepath_name, hashSize , hashSalt):
    return


class FTRL_FM(Module):

    # Follow-the-regularized-leader to Factorization Machine implementation

    def __init__(self, num_data, num_feature, option):
        super(FTRL_FM, self).__init__()


        self.num_data = num_data
        self.num_feature = num_feature

        self.task = option['task']
        self.regularizer_l1_1st = option['regularizer_l1_1st']
        self.regularizer_l2_1st = option['regularizer_l1_1st']
        self.regularizer_l1_2nd = option['regularizer_l1_2nd']
        self.regularizer_l2_2nd = option['regularizer_l2_2nd']

        self.learningrate_alpha_1st = option['learningrate_alpha_1st']
        self.learningrate_beta_1st = option['learningrate_beta_1st']
        self.learningrate_alpha_2nd = option['learningrate_alpha_2nd']
        self.learningrate_beta_2nd = option['learningrate_beta_2nd']

        self.m = option['m']

        # n : some of past gradient
        # z :
        # w: weight

        self.n_1st = np.zeros( self.num_data + 1 ,dtype = numpy_type)
        self.z_1st = np.zeros( self.num_data + 1 ,dtype = numpy_type)
        self.w_1st = np.zeros( self.num_data + 1 ,dtype = numpy_type)

        self.n_2nd = {}
        self.z_2nd = {}
        self.w_2nd = {}
        self.var_init = 0.1

    def _init_param_2nd_FM(self,current_i_user):

        if current_i_user not in self.n_2nd:
            self.n_2nd[current_i_user] = np.zeros(self.m ,dtype = numpy_type)
            self.z_2nd[current_i_user] = np.zeros(self.m ,dtype = numpy_type)
            self.w_2nd[current_i_user] = np.zeros(self.m ,dtype = numpy_type)

            for current_j in range(self.m):
                self.z_2nd[current_i_user][current_j] = np.sqrt(self.var_init)*np.random.randn()

        return

    # x : 1-hot encoding
    def _predict_reg(self, x):

        # first order 0 component
        # first order 1~ component
        for ith,x_ith in enumerate(x):
            if x_th != 0:

                if ith == 0:
                    # 1st-order update
                    self.w_1st[ith] = (-self.z_1st[ith])*\
                        1/( (self.learningrate_beta_1st + np.sqrt(self.n_1st[ith]))/self.learningrate_alpha_1st )

                    # 2nd-order update
                    self._init_param_2nd_FM(ith)
                    for current_j in range(self.m):
                        sign = 1. if self.z_2nd[ith,current_j] >= 0 else -1.
                        if np.abs(self.z_2nd[ith,current_j]) <= self.regularizer_l1_2nd :
                            self.w_2nd[ith][current_j] = 0
                        else :
                            self.w_2nd[ith][current_j] = (sign*self.regularizer_l2_1st - self.z_2nd[ith][current_j])*\
                            1/( (self.learningrate_beta_2nd + np.sqrt(self.n_2nd[ith][current_j]))/self.learningrate_alpha_2nd  + self.regularizer_l2_2nd)

                else :
                    sign = 1. if z_1st[ith] >= 0 else -1.

                    # 1st-order update
                    if np.abs(self.z_1st[ith]) <= self.regularizer_l1_1st :
                        self.w_1st[ith] = 0
                    else :
                        self.w_1st[ith] = (sign*self.regularizer_l1_1st - self.z_1st[ith])*\
                        1/( (self.learningrate_beta_1st + np.sqrt(self.n_1st[ith]))/self.learningrate_alpha_1st  + self.regularizer_l2_1st)

                    # 2nd-order update
                    self._init_param_2nd_FM(ith)
                    for current_j in range(self.m):
                        sign = 1. if self.z_2nd[ith,current_j] >= 0 else -1.
                        if np.abs(self.z_2nd[ith,current_j]) <= self.regularizer_l1_2nd :
                            self.w_2nd[ith][current_j] = 0
                        else :
                            self.w_2nd[ith][current_j] = (sign*self.regularizer_l2_1st - self.z_2nd[ith][current_j])*\
                            1/( (self.learningrate_beta_2nd + np.sqrt(self.n_2nd[ith][current_j]))/self.learningrate_alpha_2nd  + self.regularizer_l2_2nd)



        out = self.w_1st.dot(x)

        for ith in range(numel(x)):
            for jth in range(i+1,numel(x)):
                for kth in range(self.m):
                   out += self.w_2nd[x[ith]][kth]*self.w_2nd[x[jth]][kth]

        return out


    def _predict_cls(self,x_t):
        return 1./( 1. + np.exp( max( min(self._predict_reg(x_t) ,35. )  , - 35.  ))  )


    #def _dropout(self):


    def _update_param(self,x_t , p_t, y_t) :

        # 1st_order param update
        g_ith = (p_t - y_t)
        for ith, x_ith in enumerate(x):
            if x_ith != 0 :
                # 1st-order update

                sigma_ith = (1/self.learningrate_alpha_1st)*(np.sqrt(self.n_1st[ith] + g_ith**2 ) - np.sqrt(self.n_1st[ith]) )
                self.z_1st[ith] += g_ith - sigma_ith*self.w_1st[ith]
                self.n_1st[ith] += g_ith**2

        # 2nd_order param updates
        sum_gradient_2nd_order = {}
        for ith in range(x.size):
            for jth in range(x.size):
                if ith != jth :
                    for kth in range(self.m):
                        sum_gradient_2nd_order[x[ith]][kth] += self.w_2nd[x[jth]][kth]

        for ith, x_ith in enumerate(x):
            for kth in range(self.m):
                if x_ith != 0 :
                    # 1st-order update
                    g_ith_2nd = g_ith*sum_gradient_2nd_order[x[ith]][kth]
                    sigma_ith_2nd = (1/self.learningrate_alpha_2nd)*(np.sqrt(self.n_2nd[ith][kth] + g_ith_2nd**2 ) - np.sqrt(self.n_2nd[ith][kth]) )
                    self.z_2nd[ith][kth] += g_ith_2nd[ith][kth] - sigma_ith_2nd*self.w_2nd[ith][kth]
                    self.n_2nd[ith][kth] += g_ith_2nd[ith][kth]**2

        return


    def online_learning(self):

        pred_list = []
        real_list = []
        for idx in range(self.num_data):
            alpha = self.At[:, idx]

            BP_alpha = self.BT_P.t().matmul(alpha).unsqueeze(1)
            BN_alpha = self.BT_N.t().matmul(alpha).unsqueeze(1)

            scalar = self.w.t().matmul(alpha) \
                     + BP_alpha.t().matmul(BP_alpha) \
                     - BN_alpha.t().matmul(BN_alpha)

            if self.task == 'cls':
                sign_idx = self._grad_loss(scalar * self.b[idx]) * self.b[idx]
            elif self.task == 'reg':
                sign_idx = self._grad_loss(scalar - self.b[idx])
            else:
                raise NotImplementedError

            # print(self.g_w)
            self.g_w += sign_idx * alpha.unsqueeze(1)
            self.w = -self.eta * self.g_w

            self._GFD(sign_idx, alpha)

            pred_list.append(torch.tensor(scalar).double())
            real_list.append(self.b[idx])

            if idx % 100 == 0:
                print(' %d th : pred %f , real %f , loss %f ' % (
                    idx, scalar, self.b[idx], self._loss(scalar - self.b[idx])))

        return np.asarray(pred_list), np.asarray(real_list)

if __name__ == "__main__":

    nbUsers = 943
    nbMovies = 1682
    nbFeatures = nbUsers + nbMovies
    nbRatingsTrain = 90570
    nbRatingsTest = 9430

    option = {}
    option['task'] = 'cls'
    option['regularizer_l1_1st'] = 0.1
    option['regularizer_l1_1st'] = 0.1
    option['regularizer_l1_2nd'] = 0.1
    option['regularizer_l2_2nd'] = 0.1

    option['learningrate_alpha_1st'] = 0.1
    option['learningrate_beta_1st'] = 0.1
    option['learningrate_alpha_2nd'] = 0.1
    option['learningrate_beta_2nd'] = 0.1

    option['m'] = 100
    Model = FTRL_FM(nbRatingsTrain, nbFeatures , option)
    Model._init_param_2nd_FM(10)

    #print(Model.n_2nd)
    #print(Model.z_2nd)
    #print(Model.w_2nd)

    #Model.n_2nd
    #Model.n_2nd