import numpy as np
import argparse
import yaml
import data_loader
import time


import argparse
import os
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import model_trainer_nn
import model_eval_nn
import address
import data_loader

def main():
    # Data
    pickle_address = address.pickle_f()
    data_ = data_loader.data_loader_f(pickle_address)
    # ############################### arguments ###############################
    # N is batch size; D_in is input dimension (number of features);
    # H is hidden dimension; D_out is output dimension.
    # Features:  1. day_number, 2. hr_min, 3. speed,
    D_in = 36
    #
    # H_set = [100]
    # H_set2 = [20]
    # H_set3 = [0]

    # H_set = [100]
    # H_set2 = [50]
    # H_set3 = [20]

    H_set = [200]
    H_set2 = [100]
    H_set3 = [50]

    D_out = 12
    N_set = [256]
    test_every_n_epochs_set = [25]

    # alpha_set = [2, 10]
    # t_set = [2, 20]
    alpha_set = [2]
    t_set = [5]

    # 0 batch percentage means no minmax problem
    # batch_percentage_set = [95, 90, 80, 100, 70]
    batch_percentage_set = [11, 10, 30, 50, 70, 90, 100]
    # batch_percentage_set = [70]

    MSE_set = [True]

    # Multiplier of the regularizer
    weight_regul_set = [0]
    regul = '_L2_regul_'

    gpu_ = False
    # gpu_ = False

    # ############################### Training for Loop ###############################
    for_loop = [(w_regul, MSE_, H1, H2, H3, N, alpha, t, batch_percentage, test_every_n_epochs_)
                for w_regul in weight_regul_set  for MSE_ in MSE_set  for H1 in H_set  for H2 in H_set2  for H3 in H_set3
                for N in N_set  for alpha in alpha_set  for t in t_set  for batch_percentage in batch_percentage_set
                for test_every_n_epochs_ in test_every_n_epochs_set]

    for_name = ['L2 regularizer weight', 'MSE Loss', 'Size 1st Hidd. layer', 'Size 2nd Hidd. layer',
                'Size 3rd Hidd. layer', 'Batch size', 'alpha', 't', 'batch_percentage', 'test every n epochs']

    for for_temp in for_loop:
        start_time = time.time()
        w_regul, MSE_, H1, H2, H3, N, alpha, t, batch_percentage, test_every_n_epochs_ = for_temp
        print('Input dimension: %d' % D_in)
        print('Output dimension: %d' % D_out)
        for iter_setting in range(len(for_temp)):
            print(for_name[iter_setting] + ' :', for_temp[iter_setting])


        if MSE_:
            L1_loss = 1
            loss_name = 'L1'
        else:
            MSE_loss = 1
            loss_name = 'MSE'

        i = 3
        # epochs = [100]*5
        # epochs.extend([100]*2)
        # step_size = [10**(-i)]*5
        # epochs.extend([10**(-i-1)]*2)

        # epochs = [250, 250, 50, 10, 10, 10, 10]
        epochs = [5000, 3000, 1500, 50, 50, 50, 50]
        step_size = [10**(-i), 10**(-i-1), 10**(-i-2), 10**(-i-3), 10**(-i-4), 10**(-i-5), 10**(-i-6)]
        print('epochs: ', epochs)
        print('step size: ', step_size)
        print('GPU: ', gpu_)
        # Testing error epochs
        # i = 3
        # epochs = [3]
        # step_size = [10**(-i)]

        info = {'D_in': D_in, 'D_out': D_out, 'N': N, 'alpha': alpha, 't': t, 'batch_percentage': batch_percentage,
                'H1': H1, 'H2': H2, 'H3': H3, 'MSE_': MSE_, 'w_regul': w_regul, 'Epochs': epochs,
                'test_n_epochs': test_every_n_epochs_, 'step_size': step_size}

        save_add = address.model_f(info, gpu_, 'saved_models') + '.pth'
        print('Saving address of model parameters:\n', save_add)


        try:
            with open(save_add, 'r') as fh:
                # model address, information, GPU=True
                model_eval_nn.model_eval_f(save_add, info, gpu_)
        except FileNotFoundError:
            print('Start Training:')
            # model address, information, GPU=True
            # True: loss became NaN
            while model_trainer_nn.trainer_f(info, save_add, gpu_, data_):
                i += 1
                print('\nwhile loop')
                # epochs = [250, 250, 50, 10, 10, 10, 10]
                epochs = [5000, 3000, 1500, 50, 50, 50, 50]
                step_size = [10 ** (-i), 10 ** (-i - 1), 10 ** (-i - 2), 10 ** (-i - 3), 10 ** (-i - 4), 10 ** (-i - 5),
                             10 ** (-i - 6)]

                print('epochs: ', epochs)
                print('step size: ', step_size)
                # epochs = [5]
                # step_size = [10 ** (-i)]

                info = {'D_in': D_in, 'D_out': D_out, 'N': N, 'alpha': alpha, 't': t,
                        'batch_percentage': batch_percentage,
                        'H1': H1, 'H2': H2, 'H3': H3, 'MSE_': MSE_, 'w_regul': w_regul, 'Epochs': epochs,
                        'test_n_epochs': test_every_n_epochs_, 'step_size': step_size}
                if i >= 13:
                    return True

            # with open(save_add, 'r') as fh:
            #     # Load the model
            #     load_add = save_add
            #     # model address, information, GPU=True
            #     model_eval_nn.model_eval_f(load_add, info, gpu_)

        # print('Evaluation finished for:')
        # for iter_setting in range(len(for_temp)):
        #     print(for_name[iter_setting] + ' :', for_temp[iter_setting])

    # ###############################################################################
    print("---  Run time: %s minutes  ---" % int(np.ceil((time.time() - start_time)/60)))
    print('\n')
    return False


if __name__ == '__main__':
    arg = True
    while arg:
        arg = main()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config_filename', default=None, type=str,
    #                     help='Configuration filename for restoring the model.')
    # parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    # # parser.add_argument('--horizon', default=12, type=int, help='Number of horizons.')
    # args = parser.parse_args()
    # main(args)