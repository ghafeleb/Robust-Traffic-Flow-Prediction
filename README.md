# Reliable and Robust Traffic Prediction

Minimizing the average error in training the model has always been in a spotlight, i.e., finding the highest accuracy in terms of MAE, MSE, and so on. In our formulation, not only the accuracy experiences improvement but also the variance of the error has shrunk.

## Requirements
- scipy>= 1.3.1
- numpy>=1.16.1
- torch>= 1.1.0
- matplotlib>=3.1.1

You can install the requirements using the following command:
```bash
pip install -r requirements.txt
```

## Data
We use the METR-LA data from the following paper:<br>
[Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR 2018](https://arxiv.org/abs/1707.01926).<br> 
You can find the original data from the [GitHub page](https://github.com/liyaguang/DCRNN) of this paper.
The data includes the recorded speed of Los Angeles highways collected by loop sensors. It contains records of 207 sensors from March 1st 2012 to Jun 30th 2012. The location of the sensors are shown in the following figure: <br>

<p align="center">
  <img width="250" height="250" src="https://github.com/ghafeleb/TrafficPrediciton_MinMaxPercentage/blob/master/figures/METR-LA.JPG">
</p><br>
To use the METR-LA data in our model, we changed the structure of data. For example, the structure of one record in our history data before normalization is:<br>

|                     | Day number on 2012/03/01 | Average time in minutes between 02:10:00-02:15:00 | Average speed between 02:10:00-02:15:00 | Day number on 2012/03/01 | ... | Average speed between 03:05:00-03:10:00 |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Sensor x at 2012/03/01 02:10:00 |   4   |   132.5   |   70.0   |   4   |    ...   |   65.0   |

We use the information of the past one hour at the location of one sensor to predict the traffic condition in the next hour of the same location. We have twelve 5-minute time blocks for one hour. For each time block, the history data includes 3 features:
1. Day number: the number of the day in a week that starts from Monday. For instance, Monday has day number 1 and Sunday has day number 7.
2. Average time in minutes: the average time of the time interval in minutes. For example, 14:10:00-14:15:00 is ((14 * 60 + 10) + (14 * 60 + 15))/2 = 852.5 in minutes.
3. Average speed: average speed during the 5-minute block at the location of the sensor.

The label data includes the speed of each sensor for the next 1 hour. For example, the structure of one record in our label data before normalization is:<br>
|                     | Average speed between 21:00:00-21:05:00 | Average speed between 21:05:00-21:10:00 | ... | Average speed between 21:55:00-22:00:00 |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|
| Sensor x at 2012/03/01 21:00:00 |   70.0   |   68.5   | ...  |   65.0   |

The pickled file (metr_la_70_20.pkl) of the history and label data is available at [Google Drive](https://drive.google.com/drive/folders/18edZ3gsBkukyir8r0t8cCGBwWHQZs-k9?usp=sharing). The history data (X.csv) and label data (Y.csv) with CSV format are also available at the same [Google Drive](https://drive.google.com/drive/folders/18edZ3gsBkukyir8r0t8cCGBwWHQZs-k9?usp=sharing) folder. We have used min-max scaling to normalize the data between 0 and 100.

First, run the following command to create the data directories:
```bash
# Create data directories
mkdir -p data, data/pickle, data/pickle_plt, data/model
```
Next, copy the pickled data at "data/pickle" and copy two CSV files at "data".

## Train the model
You can train your model by "trainer_nn.py":
```bash
# Train the model
python trainer_nn.py
```
If your model is already trained and saved in "saved_models" directory, "model_eval_nn.py" will run the model on the train and test data.
