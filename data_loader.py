import pickle
import argparse
import os
import numpy as np
import torch


def param_loader(pickle_address):
    with open(pickle_address, 'rb') as fh:
        W = pickle.load(fh)
        return W


def param_saver(pickle_address, param):
    with open(pickle_address, "wb") as f:
        pickle.dump(param, f)


def data_loader_f(pickle_address):
    try:
        with open(pickle_address, 'r') as fh:
            # ######################################################### Train with complete metr-la
            # ######################################################### Test with downtown-la but common sensors
            my_data = pickle.load(open(pickle_address, "rb"))
            test_tens_x = my_data[0]
            test_tens_z = my_data[1]
            train_tens_x = my_data[2]
            train_tens_z = my_data[3]
            train_tens_xz = my_data[4]
            x_row_size = my_data[5]
            x_col_size = my_data[6]
            z_row_size = my_data[7]
            z_col_size = my_data[8]
            train_tens_x = torch.Tensor(train_tens_x)
            train_tens_z = torch.Tensor(train_tens_z)
            train_tens_xz = torch.Tensor(train_tens_xz)
            test_tens_x = torch.Tensor(test_tens_x)

            return test_tens_x, test_tens_z, train_tens_x, train_tens_z, train_tens_xz, x_row_size, x_col_size, z_row_size, z_col_size

    except FileNotFoundError:
        # ######################################################### Train with 80% of the metr-la
        # ######################################################### Test with 20% of the metr-la
        # ######################################################### divide the data into train and test
        # ######################################################### only metr-la for both train and test
        # metr-la size: 4833864*18 >> 3 features, 6 times, features: 1.day_number 2.hr_min 3.speed
        # csv_file_1 = 'X_10_25_hist30_NN_metr-la_train1.csv'
        csv_file_1 = 'data/X_0_60_hist60_NN_metr-la_train1.csv'

        # metr-la size: 4833864*3 >> 1 features, 6 times, features: 1.speed
        # csv_file_2 = 'Y_10_25_hist30_NN_metr-la_train1.csv'
        csv_file_2 = 'data/Y_0_60_hist60_NN_metr-la_train1.csv'

        X_data = open(csv_file_1, 'rt')
        Z_data = open(csv_file_2, 'rt')
        x = np.genfromtxt(X_data, delimiter=",")
        z = np.genfromtxt(Z_data, delimiter=",")

        x_row_size = np.size(x, 0)
        z_row_size = np.size(z, 0)

        idx = np.arange(0, x_row_size)
        np.random.shuffle(idx)
        train_perc = 0.7
        train_n = int(np.floor(x_row_size * train_perc))

        # train_tens_x = x[:train_n, :]
        # train_tens_z = z[:train_n, :]

        train_tens_x = x[idx[0:train_n], :]
        train_tens_z = z[idx[0:train_n], :]

        test_perc = 0.2
        test_n = int(np.floor(x_row_size * (train_perc + test_perc)))

        # test_tens_x = x[train_n:test_n, :]
        # test_tens_z = z[train_n:test_n, :]

        test_tens_x = x[idx[train_n:test_n], :]
        test_tens_z = z[idx[train_n:test_n], :]

        x_row_size = np.size(train_tens_x, 0)
        x_col_size = np.size(train_tens_x, 1)
        z_row_size = np.size(train_tens_z, 0)
        z_col_size = np.size(train_tens_z, 1)

        train_tens_x = torch.Tensor(train_tens_x)
        train_tens_z = torch.Tensor(train_tens_z)

        train_tens_xz = torch.cat((train_tens_x, train_tens_z), 1)

        test_tens_x = torch.Tensor(test_tens_x)

        pickle.dump([test_tens_x, test_tens_z, train_tens_x, train_tens_z, train_tens_xz, x_row_size, x_col_size,
                     z_row_size, z_col_size], open(pickle_address, "wb"))

        return test_tens_x, test_tens_z, train_tens_x, train_tens_z, train_tens_xz, x_row_size, x_col_size, z_row_size, z_col_size

