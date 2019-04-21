import torch
from torch.nn import Module
#from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

tensor_type = torch.DoubleTensor


class SFTRL_Vanila(Module):

    def __init__(self, inputs_matrix, outputs, option):

        super(SFTRL_Vanila, self).__init__()

        #self.A = inputs_matrix
        self.At = inputs_matrix.t()
        self.b = outputs
        self._thres = 1e-12

        self.num_data = inputs_matrix.shape[0]
        self.num_feature = inputs_matrix.shape[1]

        self.task = option['task']
        self.eta = option['eta']
        self.m = option['m']

        self.row_count_p = 0
        self.row_count_n = 0

        self.BT_P = tensor_type(np.zeros([self.num_feature-1, 2*self.m]))
        self.BT_N = tensor_type(np.zeros([self.num_feature-1, 2*self.m]))

        self.w = tensor_type(np.zeros([self.num_feature,1]))
        self.g_w = tensor_type(np.zeros([self.num_feature,1]))


    def _loss(self, x):

        if self.task == 'reg' :
            return x**2
        elif self.task == 'cla':
            return 1 / (1 + torch.exp(x))
        else :
            return


    def _grad_loss(self, x):

        if self.task == 'reg' :
            return 2*x
        elif self.task == 'cla' :
            return -1 / (1 + torch.exp(x))
        else :
            return


    def _predict(self):

        return

    def online_learning(self):

        pred_list = []
        real_list = []
        for idx in range(self.num_data):
            alpha = self.At[:, idx]

            BP_alpha = self.BT_P.t().matmul(alpha[:-1]).unsqueeze(1)
            BN_alpha = self.BT_N.t().matmul(alpha[:-1]).unsqueeze(1)


            scalar =  self.w.t().matmul(alpha) + BP_alpha.t().matmul(BP_alpha) - BN_alpha.t().matmul(BN_alpha)
            if self.task == 'cls':
                sign_idx = self._grad_loss(scalar * self.b[idx]) * self.b[idx]
            elif self.task == 'reg':
                sign_idx = self._grad_loss(scalar - self.b[idx])
            else:
                raise NotImplementedError


            #print(self.g_w)
            self.g_w += sign_idx*alpha.unsqueeze(1)
            self.w = -self.eta*self.g_w

            self._GFD(sign_idx, alpha[:-1])

            pred_list.append(torch.tensor(scalar).double())
            real_list.append(self.b[idx])

            if idx % 100 == 0:
                print(' %d th : pred %f , real %f , loss %f ' % (
                idx, scalar, self.b[idx], self._loss(scalar - self.b[idx])))


        return np.asarray(pred_list),np.asarray(real_list)



    def _GFD(self, sign, alpha):
        # alpha : num_feature x 1
        # sign : scalar value

        if sign <= 0:
            self.row_count_p += 1
            self.BT_P[:,self.row_count_p] = np.sqrt(-self.eta*sign)*alpha.squeeze()

            if self.row_count_p == 2*self.m - 1  :

                U, Sigma, _ = self.BT_P.t().matmul( self.BT_P).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()

                V = self.BT_P.matmul(U[:, :nnz]).matmul(  (1/Sigma[:nnz].sqrt()).diag()   )

                if nnz >= self.m:

                    self.BT_P = V[:, :self.m - 1].matmul((Sigma[:self.m - 1] - Sigma[self.m]).sqrt().diag())
                    self.BT_P = torch.cat([self.BT_P, torch.zeros([self.num_feature-1, self.m + 1]).double() ], 1)
                    self.row_count_p = self.m - 1

                else:
                    self.BT_P = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    self.BT_P= torch.cat([self.BT_P, torch.zeros([self.num_feature-1, (2 * self.m) - nnz]).double() ], 1)
                    self.row_count_p = nnz


        else:

            if self.row_count_n == 2*self.m - 1:

                self.row_count_n += 1
                self.BT_N[:,self.row_count_n] = np.sqrt(self.eta*sign)*alpha.squeeze()

                U, Sigma, _ = self.BT_N.t().matmul(self.BT_N).svd()
                Sigma[Sigma.data <= self._thres] = 0.0
                nnz = Sigma.nonzero().numel()
                V = self.BT_N.matmul(U[:, :nnz]).matmul(  (1/Sigma[:nnz].sqrt()).diag()    )

                if nnz >= self.m:
                    self.BT_N = V[:, :self.m - 1].matmul((Sigma[:self.m - 1] - Sigma[self.m]).sqrt().diag())
                    self.BT_N = torch.cat([self.BT_N, torch.zeros([self.num_feature-1, self.m + 1 ]).double() ], 1)
                    self.row_count_n = self.m - 1

                else:
                    self.BT_N = V[:, :nnz].matmul((Sigma[:nnz]).sqrt().diag())
                    self.BT_N = torch.cat([self.BT_N, torch.zeros([self.num_feature-1, (2 * self.m) - nnz] ).double()] , 1)
                    self.row_count_n = nnz

        return








if __name__ == "__main__" :

    inputs_matrix = torch.tensor(np.random.randn(5,10))
    outputs = torch.tensor(np.random.randn(5,1))
    options = {}
    options['m']  = 5
    options['eta'] = 1e-2
    options['task'] = 'reg'


    Model = SFTRL_Vanila(inputs_matrix,outputs,options)

    alpha = torch.tensor(np.random.randn(10,1))

    pred,real = Model.online_learning()

    plt.figure()
    plt.plot(pred)
    plt.plot(real)
    plt.show()