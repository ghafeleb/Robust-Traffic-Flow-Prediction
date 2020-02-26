# Relialble&RobustPrediction

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
[Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu, Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR 2018](https://arxiv.org/abs/1707.01926).<br> You can find the original data from the [GitHub page](https://github.com/liyaguang/DCRNN) of this paper.
The data includes the recorded speed of Los Angeles highways collected by loop sensors. It contains records of 207 sensors from March 1st 2012 to Jun 30th 2012.The location of sensors are shown in the following figure: <br>
<img src="https://github.com/ghafeleb/TrafficPrediciton_MinMaxPercentage/blob/master/figures/METR-LA.JPG" width="400" height="400" align="middle"><br>
To use the METR-LA data in our model, we changed the structure of data. For example, the structure of one record in our data is:<br>
|                     | day number at 02:10:00 | time in minutes at 02:10:00 | speed at 02:10:00 | day number at 02:15:00 | time in minutes at 02:15:00 | speed at 02:15:00 | ... | time in minutes at 03:05:00 | speed at 03:05:00 | day number at 03:05:00 |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Sensor x at 2012/03/01 02:10:20 |   4   |   130   |   70.0   |   4   |   135   |   68.0   |    ...   |   4   |   185   |   65.0   |

There are 3 features for 12 consecutive 5-minute time intervals in each record:
- day number: the number of the day in a week that starts from Monday. For instance, Monday has day number 1 and Sunday has day number 7.
- time in minutes: the time of the day in minutes. For example, 2:30 pm is (12+2) * 60+30 = 870 in minutes.
- speed


The pickled file of the data with the needed format in our model is available at [Google Drive](https://drive.google.com/drive/folders/18edZ3gsBkukyir8r0t8cCGBwWHQZs-k9?usp=sharing). 

## Get Started
You can train your model by "trainer_nn.py".
