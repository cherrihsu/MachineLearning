import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, D_in):
        super(Model, self).__init__()

        ##### Set actativation function #####
        self.linear_in = torch.nn.Linear(D_in, 666)
        self.linear_hid_JCTC_1 = torch.nn.Linear(666, 666)
        self.linear_hid_JCTC_2 = torch.nn.Linear(666, 666)
        self.linear_hid_JCTC_3 = torch.nn.Linear(666, 666)
        self.linear_hid_JCTC_4 = torch.nn.Linear(666, 666)
        self.linear_out = torch.nn.Linear(666, 1)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self.batchnorm = torch.nn.BatchNorm1d(D_in)
        self.batchnorm_JCTC_1 = torch.nn.BatchNorm1d(666)
        self.batchnorm_JCTC_2 = torch.nn.BatchNorm1d(666)
        self.batchnorm_JCTC_3 = torch.nn.BatchNorm1d(666)
        self.batchnorm_JCTC_4 = torch.nn.BatchNorm1d(666)


    def forward(self, x):

        ##### Input Layer #####
        H1 = self.leaky_relu(self.linear_in(self.batchnorm(x)))

        ##### Hidden Layer #####      
        H2 = self.leaky_relu(self.linear_hid_JCTC_1(self.batchnorm_JCTC_1(H1)))
        H3 = self.leaky_relu(self.linear_hid_JCTC_2(self.batchnorm_JCTC_2(H2)))
        H4 = self.leaky_relu(self.linear_hid_JCTC_3(self.batchnorm_JCTC_3(H3)))
        H5 = self.leaky_relu(self.linear_hid_JCTC_4(self.batchnorm_JCTC_4(H4)))

        ##### Output Layer #####
        y_pred = self.linear_out(H5)
        return y_pred
