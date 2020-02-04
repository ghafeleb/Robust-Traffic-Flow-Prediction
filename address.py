import os


def pickle_f():
    folder_ = 'data\\pickle'
    name_ = 'metr_la_70_20.pkl'
    add_ = os.path.join(folder_, name_)
    return add_


def pickle_f_plt():
    folder_ = 'data\\pickle_plt'
    name_ = 'metr_la_70_20.pkl'
    add_ = os.path.join(folder_, name_)
    return add_


def model_f(info, gpu_, folder_):
    # N is batch size; D_in is input dimension (number of features);
    # H is hidden dimension; D_out is output dimension.
    D_in, H1, H2, H3, D_out, N, alpha, t = info['D_in'], info['H1'], info['H2'], info['H3'], info['D_out'], info['N'], info['alpha'], info['t']
    # 0 batch percentage means no minmax problem
    batch_percentage = info['batch_percentage']
    MSE_ = info['MSE_']
    # Multiplier of the regularizer
    w_regul = info['w_regul']
    regul = '_L2_regul_'
    test_n_epochs = info['test_n_epochs']
    if MSE_:
        loss_name = 'MSE'
    else:
        loss_name = 'L1'

    if gpu_:
        processor = '_gpu_'
    else:
        processor = '_cpu_'
    name_1 = 'NN_minmax_metrla_70_20_hist1_12_pred1_12' + regul + str(w_regul) + '_Loss_Norm_' + loss_name\
             + '_max_' + str(batch_percentage) + '_N_' + str(N) + '_D_in_' + str(D_in) + '_alpha_' + str(alpha) + \
             '_t_' + str(t) + '_test_n_epochs_' + str(test_n_epochs) + processor
    if H2 == 0:
        name_ = name_1 + '_H1_' + str(H1)
    elif H3 == 0:
        name_ = name_1 + '_H1_' + str(H1) + '_H2_' + str(H2)
    else:
        name_ = name_1 + '_H1_' + str(H1) + '_H2_' + str(H2) + '_H3_' + str(H3)

    add_ = os.path.join(folder_, name_)
    return add_



