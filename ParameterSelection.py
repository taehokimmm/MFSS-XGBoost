import os
import pandas as pd
import matplotlib.pyplot as plt
from StockHelper import Stock, StockPile, createICDataFrame, FACTORS
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

plt.rcParams["font.family"] = "SF Pro Display"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

TRAIN_PERIOD = pd.date_range("2011-01-05", "2018-12-31", freq="10D")
TEST_PERIOD = pd.date_range("2019-06-30", "2019-12-31", freq="10D")
BASELINE_DATE = pd.to_datetime("2002-05-21")

stocks = ["AAPL", "AXP", "AMGN", "BA", "CAT", "CVX", "CSCO", "GS"]
djia = pd.read_csv(os.path.join(DATA_DIR, "DJIA.csv"), parse_dates=["Date"])
dow30 = StockPile("DOW30", djia)

for stock in stocks:
    df_price = pd.read_csv(os.path.join(DATA_DIR, f"{stock}.csv"), parse_dates=["Date"])
    df_stats = pd.read_csv(
        os.path.join(DATA_DIR, f"{stock}_STAT.csv"), parse_dates=["Date"]
    )
    dow30.addStock(Stock(stock, df_price, df_stats))

# Create the training data
IC_train, nextIC_train, _ = createICDataFrame(dow30, TRAIN_PERIOD)
IC_test, nextIC_test, _ = createICDataFrame(dow30, TEST_PERIOD)

xgbModel_test = MultiOutputRegressor(XGBRegressor())
gridTree = GridSearchCV(
    xgbModel_test,
    param_grid={
        "estimator__n_estimators": [100, 200],
        "estimator__learning_rate": [0.05],
        "estimator__max_depth": [3, 4, 5, 6],
        "estimator__subsample": [0.2, 0.4, 0.6],
        "estimator__colsample_bytree": [0.2, 0.4, 0.6, 0.8, 1],
        "estimator__reg_lambda": [10, 20],
        "estimator__gamma": [10],
        "estimator__min_child_weight": [1, 2, 3],
        "estimator__colsample_bylevel": [0.6, 0.8, 1],
    },
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=2,
)
gridTree.fit(IC_train, nextIC_train)
print(gridTree.best_params_)
print(gridTree.best_score_)
print(gridTree.best_estimator_)
print(gridTree.score(IC_test, nextIC_test))
optim = gridTree.best_estimator_

# Graph Plot
plt.figure(figsize=(18, 8))
plt.suptitle("IC Weights over testing period (GridSearchCV)", fontsize=20)
IC_predict = pd.DataFrame(optim.predict(IC_test), columns=FACTORS)
for i in range(len(FACTORS)):
    plt.subplot(2, 3, i + 1)
    plt.title(FACTORS[i], fontsize=16)
    plt.plot(IC_predict.iloc[:, i], label="prediction")
    plt.plot(nextIC_test.iloc[:, i], label="real")
    plt.legend()

plt.savefig(dpi=300, fname="prediction.png")
plt.show()
