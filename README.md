# MinMaxPercentage

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
It includes the data collected by Los Angeles loop sensors on highways. The location of sensors are shown in the following figure from his paper. <br>
<img src="https://github.com/ghafeleb/TrafficPrediciton_MinMaxPercentage/blob/master/figures/METR-LA.JPG" width="400" height="400" align="middle"><br>
To use this data with the needed format in our model, you can find the pickled file [here](https://drive.google.com/drive/folders/18edZ3gsBkukyir8r0t8cCGBwWHQZs-k9?usp=sharing). You can also find the original data on the [GitHub](https://github.com/liyaguang/DCRNN) page of the Li's paper. 

## Get Started
You can train your model by "trainer_nn.py".
