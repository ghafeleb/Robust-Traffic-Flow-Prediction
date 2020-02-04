import matplotlib.pyplot as plt
import numpy as np

def plot_perc(arr, info, perc_list):
    arr_asc = np.sort(arr)
    arr_desc = arr_asc[::-1]
    arr_row_size = np.size(arr_desc, 0)
    batch_percentage = info['batch_percentage']
    print('Batch percentage: %.2f' %batch_percentage)
    for perc_ in perc_list:
        perc_size = int(arr_row_size * perc_)
        err_temp = np.mean(arr_desc[:perc_size])
        perc_sign = '%'
        print('Mean of first %.2f%c error: %.10f' %(perc_*100, perc_sign, err_temp))
