import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from StockHelper import Stock, StockPile, FACTORS
from StockHelper import createICDataFrame
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
from itertools import combinations

import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

TRAIN_PERIOD = pd.date_range("2011-01-05", "2018-12-31", freq="10D")
TEST_PERIOD = pd.date_range("2019-06-27", "2019-12-31", freq="10D")
BASELINE_DATE = pd.to_datetime("2002-05-21")

_CONFIG = ["3T", "5T", "10T", "1F", "3F", "5F"]

stocks = ["AAPL", "AXP", "AMGN", "BA", "CAT", "CVX", "CSCO", "GS"]
djia = pd.read_csv(os.path.join(DATA_DIR, "DJIA.csv"), parse_dates=["Date"])
dow30 = StockPile("DOW30", djia)
for stock in stocks:
    df_price = pd.read_csv(os.path.join(DATA_DIR, f"{stock}.csv"), parse_dates=["Date"])
    df_stats = pd.read_csv(
        os.path.join(DATA_DIR, f"{stock}_STAT.csv"), parse_dates=["Date"]
    )
    dow30.addStock(Stock(stock, df_price, df_stats))

bestConfig = []
bestScore = -1
for L in range(2, len(_CONFIG) + 1):
    for _config in combinations(_CONFIG, L):
        config = []
        now = datetime.now()
        print("[{}] config:{}".format(now.strftime("%Y%m%d-%H:%M"), _config))
        for tmp in _config:
            if tmp[-1] == "T":
                for i in [1, 3, 5, 10, 20, 30]:
                    config.append([True, i, int(tmp[0])])
            else:
                for i in [1, 3, 5, 10, 20, 30]:
                    if i >= int(tmp[0]):
                        config.append([False, i, int(tmp[0])])

        IC_train, nextIC_train, _ = createICDataFrame(dow30, TRAIN_PERIOD, config)
        IC_test, nextIC_test, _ = createICDataFrame(dow30, TEST_PERIOD, config)

        xgbModel = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_child_weight=1,
                gamma=0,
                subsample=0.4,
                colsample_bylevel=0.6,
                colsample_bytree=0.4,
                reg_lambda=10,
                tree_method="hist",
            )
        )
        xgbModel.fit(IC_train, nextIC_train)
        score1 = xgbModel.score(IC_train, nextIC_train)
        score2 = xgbModel.score(IC_test, nextIC_test)
        now = datetime.now()
        print("[{}] score1 : {} score2 : {}".format(now, score1, score2))
        if score2 > bestScore:
            bestScore = score2
            bestConfig = _config

print("best config:{}".format(bestConfig))
print("best score:{}".format(bestScore))
