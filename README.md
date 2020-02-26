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
The METR-LA data is from the paper by Yaguang. It includes the data collected by Los Angeles loop sensors on highways. The location of sensors are shown in the following figure from his paper. 
![METR-LA](figures/METR-LA.JPG "METR-LA" =250x250)
You can also find the original data on his [GitHub](https://github.com/liyaguang/DCRNN). To use this data with the needed format in our Neural Network, you can find the pickled file [here](https://drive.google.com/drive/folders/18edZ3gsBkukyir8r0t8cCGBwWHQZs-k9?usp=sharing).
