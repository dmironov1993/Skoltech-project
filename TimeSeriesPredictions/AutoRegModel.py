#https://www.kaggle.com/c/ozonmasters-2022-ml-2-contest-1/leaderboard?

#Score: 3.34795
#Public score: 3.37274


import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter('ignore')


def smape(true, pred):
    return 2 * np.mean(np.abs(pred - true) / (np.abs(true) + np.abs(pred))) * 100
  
  
df_train = pd.read_csv('train.csv', index_col=False)

best_params = {}

for k in tqdm(range(3600)):
    y = df_train[df_train['item_id'] == k].value.values
    y_train, y_val = y[:int(y.size * 0.8)], y[int(y.size * 0.8):]
    best_lag = 24
    best_period = 24
    best_score = 2000
    if y_train.size > 150:
        for lags in range(12,49,2):
            for period in range(12,49,2):
                res = AutoReg(y_train, lags=lags, trend='ct', seasonal=True, period=period).fit()
                y_hat = res.predict(start=int(y.size * 0.8), end=int(y.size)-1)
                score = smape(y_val, y_hat)
                if score < best_score:
                    best_lag = lags
                    best_period = period
                    best_score = score
    best_params[k] = {'best_lag':best_lag, 'best_period':best_period, 'best_smape':best_score}
    

df_test = pd.read_csv('test.csv', index_col=False)
df_submission = pd.read_csv('submission.csv', index_col=False)


real_preds = []
real_all_preds = []
for k in tqdm(range(3600)):
    y = df_train[df_train['item_id']==k]['value'].values
    df = df_test[df_test['item_id'] == k]['vals_id'].values
    res = AutoReg(y, 
                lags=best_params[k]['best_lag'], 
                trend='ct', 
                seasonal=True,
                period=best_params[k]['best_period']).fit()
    yhat = res.predict(start=df[0], end=df[-1])
    real_preds.append(yhat)
    real_all_preds.extend(yhat)
    
df_submission['value'] = real_all_preds
df_submission.to_csv('solution.csv', index=False)
