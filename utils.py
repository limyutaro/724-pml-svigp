import numpy as np
import pandas as pd
import torch

# import positivity constraint
from gpytorch.kernels import Kernel, PeriodicKernel, LinearKernel, RBFKernel, RQKernel, ScaleKernel, ProductKernel, AdditiveKernel

def customkernel():
    #### long term
    k_long_term_RBF = RBFKernel()
    k_long_term_RBF.lengthscale = 67.0

    k_long_term = ScaleKernel(k_long_term_RBF)
    k_long_term.outputscale = 66.0

    #### seasonal
    k_seasonal_RBF = RBFKernel()
    k_seasonal_RBF.lengthscale = 90.0

    k_seasonal_periodic = PeriodicKernel()
    k_seasonal_periodic.period_length = 1.0
    # this must be squared, see https://github.com/cornellius-gp/gpytorch/issues/1020
    k_seasonal_periodic.lengthscale = 4.3

    k_seasonal = ScaleKernel(k_seasonal_RBF * k_seasonal_periodic)
    k_seasonal.outputscale = 2.4

    ### medium term
    k_medium_RQ = RQKernel()
    k_medium_RQ.alpha = 1.2
    k_medium_RQ.lengthscale = 0.78

    k_medium = ScaleKernel(k_medium_RQ)
    k_medium.outputscale = 0.66

    #### noise term
    k_noise_RBF = RBFKernel()
    k_noise_RBF.lengthscale = 1.6

    k_noise = ScaleKernel(k_noise_RBF)
    k_noise.outputscale = 0.18

    return k_long_term + k_seasonal + k_medium + k_noise


from sklearn.datasets import fetch_openml
def loadco2(test_size=0.3, subsample_rate=1):
    co2 = fetch_openml(data_id=41187, as_frame=True)
    co2_data = co2.frame
    co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
    co2_data = co2_data[["date", "co2"]].set_index("date")
    co2_data = co2_data.dropna()

    x = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
    y = co2_data["co2"].to_numpy()
    
    x, y = x[::subsample_rate], y[::subsample_rate]
    
    cut_of_idx = int((1-test_size)*len(x))
    
    train_x, train_y = torch.Tensor(x[:cut_of_idx]), torch.Tensor(y[:cut_of_idx])
    test_x, test_y = torch.Tensor(x[cut_of_idx:]), torch.Tensor(y[cut_of_idx:])
    
    # train_x = 0.2*(train_x - 1950)
    # test_x = 0.2*(test_x - 1950)
    train_y = 0.2*(train_y - 300)
    test_y = 0.2*(test_y - 300)

    return train_x, train_y, test_x, test_y