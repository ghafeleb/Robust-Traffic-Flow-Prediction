import time
start_time = time.time()

import data_loader
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable as V
import address
import torch.nn.functional as F
import data_loader
import model_eval_nn

# cuda_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(cuda_)

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
class SimpleRegression(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(SimpleRegression,self).__init__()
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

def trainer_f(info, save_add, gpu_, data_):

    # N is batch size; D_in is input dimension (number of features);
    # H is hidden dimension; D_out is output dimension.
    D_in, H1, H2, H3, D_out, N, alpha, t = info['D_in'], info['H1'], info['H2'], info['H3'], \
                    info['D_out'], info['N'], info['alpha'], info['t']
    # 0 batch percentage means no minmax problem
    batch_percentage = info['batch_percentage']
    test_n_epochs = info['test_n_epochs']
    MSE_ = info['MSE_']

    # Multiplier of the regularizer
    w_regul = info['w_regul']
    regul = '_L2_regul_'

    Epochs = info['Epochs']
    step_size = info['step_size']

    L1_loss, MSE_loss= 0, 0
    if MSE_:
        MSE_loss = 1
        loss_name = 'MSE'
    else:
        L1_loss = 1
        loss_name = 'L1'

    # ############################### Loading Data ###############################
    # pickle_address = address.pickle_f()
    # test_x, test_y, train_x, train_y, train_xy, x_row_size, x_col_size, y_row_size, y_col_size = \
    #     data_loader.data_loader_f(pickle_address)
    test_x, test_y, train_x, train_y, train_xy, x_row_size, x_col_size, y_row_size, y_col_size = data_

    k = int(batch_percentage * x_row_size)
    n = int(x_row_size)

    print('Training data shape: ', np.shape(train_x))
    print('Testing data shape: ', np.shape(test_x))

    dataloader = DataLoader(train_xy, batch_size=N, shuffle=True, num_workers=0)
    # ###############################################################
    # # Choose the number of layers
    # model = TwoLayerNet(D_in, H1, D_out)

    # dtype = torch.float
    # if gpu_: # GPU
    #     device = torch.device("cuda:0")
    # else: # CPU
    #     device = torch.device("cpu")

    # GPU or CPU
    if gpu_:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if H1 == 0:
        model = SimpleRegression(D_in, D_out)
    if H2 == 0:
        model = TwoLayerNet(D_in, H1, D_out)
    elif H3 == 0:
        model = ThreeLayerNet(D_in, H1, H2, D_out)
    elif H3 > 0:
        model = FourLayerNet(D_in, H1, H2, H3, D_out)

    model.type(dtype)
    model.train()

    y_plot = np.array([])
    counter = 0

    alpha = torch.tensor([alpha]).type(dtype)
    mu_ = torch.randn(1).type(dtype).requires_grad_()

    loss_checker = float('inf')
    improv = 0
    for iter_lr in range(np.size(Epochs)):
        for iter_epoch in range(Epochs[iter_lr]):
            lr_ = step_size[iter_lr]

            # optimizer = torch.optim.SGD(var_list, lr=lr_, momentum=0.9, nesterov=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_)

            # weight_decay: multiplier of L2 regularizer
            # optimizer = torch.optim.Adam(var_list, lr=lr_, weight_decay=w_regul)

            counter += 1

            y_plot0 = 0

            start_time = time.time()
            for batch_num, sample_batched_ in enumerate(dataloader):
                sample_batched = sample_batched_.type(dtype)
                # sample_batched = sample_batched_
                size_sample_batched = sample_batched.size(0)
                # lambda_ = torch.randn(size_sample_batched, 1).type(dtype).requires_grad_()
                # ##############################################################################
                # 1
                # print('mu_:', mu_.size())
                # print('alpha:', alpha.size())
                const_ = torch.log(1 / (1 + alpha))
                # print('const_:', const_.size())
                # size of y_pred: D_out
                y_pred = model(sample_batched[:, 0:D_in])
                # print('y_pred:', y_pred.size())
                # Size of y_: batch size * size of input (12)
                # if L1_loss == 1:
                #     criterion = torch.nn.L1Loss(reduction='mean')
                # elif MSE_loss == 1:
                #     criterion = torch.nn.MSELoss(reduction='mean')
                f_ = (y_pred - sample_batched[:, D_in:D_in + y_col_size]).pow(2).sum(dim=1)
                # print('f_:', f_.size())
                # f_ = criterion(y_pred[:, :], sample_batched[:, D_in:D_in + y_col_size])
                lambda_ = mu_ - (1/t) * const_ - f_
                # print('lambda_:', lambda_.size())
                loss = k / x_row_size * mu_ + lambda_ + (alpha / t) * F.softplus(t * (f_ - mu_ - lambda_), beta=1, threshold=10)
                # print('loss:', loss.size())

                loss = loss.sum()
                # print('loss.sum():', loss.size())

                # print('Max loss for iteration ' + str(counter) + ' and batch_num ' + str(batch_num + 1) +
                #       ' with size ' + str(size_sample_batched) + ': ' + str(loss.item()))

                temp_loss_value = np.array(loss.item())
                if np.isnan(temp_loss_value):
                    print('Loss is NaN!')
                    return True
                y_plot0 += temp_loss_value

                # avg_loss_ = temp_loss_value / np.size(sample_batched, 0)
                #
                # print('Loss at epoch ' + str(counter) + ' with batch size of ' + str(N) +
                #       ' and step size ' + str(lr_) + ': ' + str(avg_loss_))

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # avg_loss_ = y_plot0 / np.size(train_xy, 0)
            # print('Loss at epoch ' + str(counter) + ' with batch size of ' + str(N) +
            #       ' and step size ' + str(lr_) + ': ' + str(avg_loss_))

            # improv checks the number of iterations that we have not experienced improvement in loss value
            improv += 1

            # avg_loss = y_plot0/(batch_num + 1)
            avg_loss = y_plot0/(size_sample_batched)

            print('Average loss of batches at epoch ' + str(counter) + ' with batch size of ' + str(N) +
                  ' and step size ' + str(lr_) + ': ' + str(avg_loss))


            # print("---  Run time: %s seconds  ---" % (time.time() - start_time))
            # print('y_plot0/size_sample_batched', y_plot0/(batch_num + 1))
            #
            # if avg_loss >= 1100000:
            #     if (iter_epoch+1) >= test_n_epochs:
            #         return True

            if avg_loss < 1000000:
                y_plot = np.hstack((y_plot, np.array([avg_loss])))
                # print('y_plot: ... ', y_plot)

            # If the min loss so far is greater than (y_plot0/(batch_num + 1)) + 0.01, we will put the improv = 0
            # It means that we should wait at least test_n_epochs more epochs to change the step size
            if loss_checker > avg_loss:
                loss_checker = avg_loss
                improv = 0

            # If test_every_n_epochs number of epochs, we have not experienced any improvement, we decrease the step size
            if improv >= (test_n_epochs):
                improv = 0
                # Save the NN model
                torch.save(model, save_add)
                break

            # Save the NN model
            torch.save(model, save_add)
        # Save the NN model
        torch.save(model, save_add)
    # Save the NN model
    torch.save(model, save_add)
    # # Plot the error
    # from matplotlib.pyplot import figure
    # figure(num=None, figsize=(15, 15), dpi=100, facecolor='w', edgecolor='k')
    # plt.rc('xtick', labelsize=14)
    # plt.rc('ytick', labelsize=14)
    # plt.title('NN, METR-LA, Loss:' + loss_name + ', batch size=' + str(N) +
    #           ', Hidden layer=' + str(H1) + ', Hidden layer2=' + str(
    #           H2) + ', Hidden layer3=' + str(H3) +
    #           ', Dim_output=' + str(D_out) +
    #           ', batch_percentage=' + str(batch_percentage), size=14)
    # plt.xlabel('Iteration', size=14)
    # plt.ylabel('Error', size=14)
    # plt.plot(y_plot)
    # plt.show()

    y_plot = list(y_plot)
    # print('y_plot:', y_plot)

    # Save the information of plot
    save_add = address.model_f(info, gpu_, 'data\\pickle_plt') + '_plt.pth'
    data_loader.param_saver(save_add, y_plot)
    print("---  Run time: %s minutes  ---" % int(np.ceil((time.time() - start_time)/60)))

    return False


