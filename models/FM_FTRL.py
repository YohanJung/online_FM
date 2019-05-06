import numpy
import torch
from torch.autograd import Variable
from torch.nn import Module
from models.FM_Base import FM_Base

import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

tensor_type = torch.DoubleTensor
numpy_type = np.float64


# Factorization Machines + FTRL


class FM_FTRL(FM_Base):

    # Follow-the-regularized-leader to Factorization Machine implementation
    def __init__(self, inputs_matrix, outputs, option):

        super(FM_FTRL, self).__init__(inputs_matrix, outputs, option)


        self._init_parameter()


    def _init_parameter(self):
        self.w1 = Variable(torch.randn(self.num_feature,1).type(tensor_type))
        self.W2 = Variable(torch.randn(2*self.m,self.num_feature-1).type(tensor_type) , requires_grad = True)


    def online_learning(self):

        pred_list = []
        real_list = []
        g_w1 = torch.zeros_like(self.w1)
        g_W2 = torch.zeros_like(self.W2)

        for idx in range(self.num_data):

            alpha = self.At[:, idx].unsqueeze(1)
            temp_scalar = self.W2.matmul(alpha[:-1])
            scalar = self._predict( self.w1.t().matmul(alpha) + temp_scalar.t().matmul(temp_scalar) )


            if self.task == 'cls':
                sign_idx = self._grad_loss(scalar * self.b[idx]) * self.b[idx]
            elif self.task == 'reg':
                sign_idx = self._grad_loss(scalar - self.b[idx])
            else:
                raise NotImplementedError

            g_w1 += sign_idx*alpha
            g_W2 += 2*self.W2.matmul(alpha[:-1]).matmul(alpha[:-1].t())

            self.w1 = -self.eta*g_w1
            self.W2 = -self.eta*g_W2

            pred_list.append(torch.tensor(scalar).squeeze().double())
            real_list.append(self.b[idx])

            if idx % 100 == 0:
                # print(' %d th : pred %f , real %f , loss %f ' % (
                # idx, scalar, self.b[idx], self._loss(scalar - self.b[idx])))
                print(' %d th : pred %f , real %f ' % (idx, scalar, self.b[idx], ))


        # return torch.tensor(pred_list).reshape([-1,1]).double(),\
        #        torch.tensor(np.asarray(real_list).reshape[-1,1]).double()

        return np.asarray(pred_list),np.asarray(real_list)


