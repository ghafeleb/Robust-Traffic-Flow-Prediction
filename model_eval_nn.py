import numpy as np
import argparse
import yaml
import data_loader

print('\n')

import argparse
import os
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import address
import plot_

# cuda_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(cuda_)
class SimpleRegression(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(SimpleRegression, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        y_prediction = self.linear1(x)
        return y_prediction

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_prediction = self.linear2(h_relu)

        return y_prediction



class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu).clamp(min=0)
        y_prediction = self.linear3(h_relu2)

        return y_prediction


class FourLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, H2, H3, D_out):
        super(FourLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu).clamp(min=0)
        h_relu3 = self.linear3(h_relu2).clamp(min=0)
        y_prediction = self.linear4(h_relu3)

        return y_prediction

def model_eval_f(load_add, info, gpu_):
    # N is batch size; D_in is input dimension (number of features);
    # H is hidden dimension; D_out is output dimension.
    D_in = info['D_in']
    H1 = info['H1']
    H2 = info['H2']
    H3 = info['H3']
    D_out = info['D_out']
    N = info['N']

    # 0 batch percentage means no minmax problem
    batch_percentage = info['batch_percentage']

    # Loss function, L1_loss = 1, MSE_loss = 2
    MSE_ = info['MSE_']

    if MSE_:
        L1_loss = 1
        loss_name = 'L1'
    else:
        MSE_loss = 1
        loss_name = 'MSE'

    # Multiplier of the regularizer
    w_regul = info['w_regul']
    regul = '_L2_regul_'

    # GPU or CPU
    if gpu_:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # ############################### Loading Data ###############################
    pickle_address = address.pickle_f()
    test_x, test_y, train_x, train_y, train_xy, x_row_size, x_col_size, y_row_size, y_col_size = \
        data_loader.data_loader_f(pickle_address)

    print('Training data x shape: ', np.shape(train_x))
    print('Testing data x shape: ', np.shape(test_x))
    print('Training data y shape: ', np.shape(train_y))
    print('Testing data y shape: ', np.shape(test_y))

    # ############################# Loading Error plot ##############################
    save_add = address.model_f(info, gpu_, 'data\\pickle_plt') + '_plt.pth'
    y_plot = data_loader.param_loader(save_add)
    y_plot = np.array(y_plot)
    # print('Error array:\n', y_plot)
    # Plot the error
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(15, 15), dpi=100, facecolor='w', edgecolor='k')
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.title('NN, METR-LA, Loss:' + loss_name + ', batch size=' + str(N) +
              ', Hidden layer=' + str(H1) + ', Hidden layer2=' + str(
              H2) + ', Hidden layer3=' + str(H3) +
              ', Dim_output=' + str(D_out) +
              ', batch_percentage=' + str(batch_percentage), size=14)
    plt.xlabel('Iteration', size=14)
    plt.ylabel('Error', size=14)
    plt.plot(y_plot)
    plt.show()

    # ############################## Preparing Model #################################
    model = torch.load(load_add)
    model.eval()
    # Prediction of speed
    speed_pred = model(test_x.type(dtype))
    # Tensor to numpy array
    speed_pred = speed_pred.data.cpu().numpy()

    # ###########################################################################
    # Predicted next 10-25 minutes
    speed_pred = speed_pred
    speed_pred_flat = np.reshape(speed_pred, (-1,))
    # ###########################################################################
    # Real next 10-25 minutes
    speed_real = test_y

    # Reshape to an array for every 15 minutes
    # In the following arrays we have this sequence >> 10-25, 10-25, 10-25, ....
    speed_real_flat = np.reshape(speed_real, (-1,))
    # ######################################################################################
    # if mph is 1, results are scaled to 0-70
    # if mph is 0, results are scaled to 0-100
    mph = 1
    if mph == 0:
        mph_or_not = 'not '
        mph_coef = 1
    elif mph == 1:
        mph_or_not = ''
        mph_coef = 70 / 100

        # Average of error for each  array of prediction
        pred_avg_err_mph0 = np.linalg.norm(speed_pred - speed_real, ord=1,
                                           axis=1) / np.size(speed_real, 1)

        # NN
        # Results based on all errors for all sensors and time intervals
        print('NN MinMax, Results based on all errors:')
        print('Mean of all errors (' + mph_or_not + 'mph)= ',
              np.mean(pred_avg_err_mph0, axis=0) * mph_coef)
        print('Standard Deviation of all errors (' + mph_or_not + 'mph)= ',
              np.sqrt(np.var(pred_avg_err_mph0)) * mph_coef)
        print('Mean + Standard Deviation of all errors (' + mph_or_not + 'mph)= ',
              (np.mean(pred_avg_err_mph0, axis=0) + np.sqrt(np.var(pred_avg_err_mph0))) * mph_coef)

        perc_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        plot_.plot_perc(pred_avg_err_mph0 * mph_coef, info, perc_list)
    # #######################################################################
    # Assign labels 1, 0, and -1
    # y = prediction, x = real
    # y - x
    speed_diff = speed_pred_flat - speed_real_flat
    # y - x > a   -->   label +1
    # y - x < b   -->   label -1
    # otherwise   -->   label 0
    a = 10
    b = -10

    idx_speed_label_1 = speed_diff > a
    idx_speed_label_neg1 = speed_diff < b

    # Assign labels
    speed_label = np.zeros((np.size(speed_real_flat)))

    speed_label[idx_speed_label_1] = 1
    speed_label[idx_speed_label_neg1] = -1

    speed_label = speed_label.astype(int)

    unique_speed_labels_hashnet_april, counts_speed_labels_hashnet_april = np.unique(
        speed_label, return_counts=True)

    # print('Label -1 percentage for speed, march, Hash-net algorithm, 10-25 = ',
    #       counts_speed_labels_hashnet_april[0] / np.sum(counts_speed_labels_hashnet_april) * 100)
    print('Label 0 percentage for speed = ',
          counts_speed_labels_hashnet_april[1] / np.sum(counts_speed_labels_hashnet_april) * 100)
    # print('N=' + str(N) + ', D_in=' + str(D_in) + ', H1=' + str(H1) + ', D_out=' + str(D_out))


