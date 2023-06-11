# %%
import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "SF Pro Display"

# %%
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

TRAIN_PERIOD = pd.date_range("2011-01-05", "2017-12-31", freq="10D")
TEST_PERIOD = pd.date_range("2018-05-01", "2019-12-31", freq="10D")
BASELINE_DATE = pd.to_datetime("2002-05-21")

CONFIG = [20, 1, 5, 10, 30]


# %%
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

FACTORS = [
    "valuation",
    "profitability",
    "growth",
    "quality",
    "liquidity",
    "momentum_and_reversal",
]


class Stock:
    def __init__(self, name, df_price: pd.DataFrame, df_stats: pd.DataFrame):
        self.name = name
        self.df = df_price
        self.dfStatRaw = df_stats.sort_values(by="Date")
        self.dfStat = self.interpolateStats(self.dfStatRaw)

    def __str__(self):
        return self.name

    # Calculate the factor score for a given time
    def calcFactor(self, factor, currentTime):
        dfTime = self.df[self.df["Date"] <= pd.to_datetime(currentTime)].tail(1)
        dfStatTime = self.dfStat[
            self.dfStat["Date"] <= pd.to_datetime(currentTime)
        ].tail(1)
        if factor == "size":
            return dfStatTime.iloc[0]["SIZE"]
        elif factor == "valuation":
            return dfStatTime.iloc[0]["PBR"]
        elif factor == "profitability":
            return dfStatTime.iloc[0]["ROE"]
        elif factor == "growth":
            return dfStatTime.iloc[0]["GROWTH"]
        elif factor == "quality":
            return dfStatTime.iloc[0]["Current Ratio"]
        elif factor == "liquidity":
            return dfTime.iloc[0]["ILLIQ"]
        elif factor == "momentum_and_reversal":
            return self.calcReturn(currentTime, 60)
        else:
            return None

    # Calculate the return for a given time
    def calcReturn(self, currentTime, timeFrame):
        now = (
            self.df[self.df["Date"] <= pd.to_datetime(currentTime)]
            .tail(1)
            .iloc[0]["Close"]
        )
        next = (
            self.df[
                self.df["Date"]
                <= pd.to_datetime(currentTime) + pd.Timedelta(timeFrame, unit="d")
            ]
            .tail(1)
            .iloc[0]["Close"]
        )
        return (next - now) * 100 / now

    # Interpolate the missing values in the stats dataframe
    def interpolateStats(self, df: pd.DataFrame):
        df = df.set_index("Date")
        try:
            df = df.reindex(
                pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq="D",
                )
            )
        except:
            print(self.name)
            print(df.index.min(), df.index.max())
            print(df)
            sys.exit()
        df = df.interpolate(method="cubicspline", axis=0)
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})
        return df


class Indicator:
    def __init__(self, name, df: pd.DataFrame):
        self.name = name
        self.df = df

    def __str__(self):
        return self.name

    def calcIndicator(self, currentTime):
        return (
            self.df[self.df["DATE"] <= pd.to_datetime(currentTime)].tail(1).iloc[0, 1]
        )

    def calcYoY(self, currentTime):
        now = self.calcIndicator(currentTime)
        lastYear = self.calcIndicator(currentTime - pd.Timedelta(365, unit="d"))
        return (now - lastYear) * 100 / lastYear


class StockPile:
    def __init__(self, name="StockPile", index: pd.DataFrame = None):
        self.name = name
        self.stocks = []
        self.indicators = []
        self.index = index

    def __str__(self):
        return self.name

    def addStock(self, stock: Stock):
        self.stocks.append(stock)

    def addIndicator(self, indicator):
        self.indicators.append(indicator)

    def getIndex(self, currentTime):
        return (
            self.index[self.index["Date"] <= pd.to_datetime(currentTime)]
            .tail(1)
            .iloc[0]["Close"]
        )

    def getIndicators(self, currentTime):
        return (
            [i.calcIndicator(currentTime) for i in self.indicators]
            + [i.calcYoY(currentTime) for i in self.indicators]
            + [self.getIndex(currentTime)]
        )

    # Calculate the IC for a given factor in a given time
    def calcIC(self, factor, currentTime, timeFrame=10, verbose=False):
        fact = []
        alpha = []
        for stock in self.stocks:
            try:
                _fact = stock.calcFactor(factor, currentTime)
            except:
                continue
            if not np.isnan(_fact):
                fact.append(_fact)
                alpha.append(stock.calcReturn(currentTime, timeFrame))
        if verbose:
            print(fact)
            print(alpha)
            print(len(fact), len(alpha))

        return np.corrcoef(fact, alpha)[0, 1]

    def showICScatter(self, currentTime, timeFrame=10, verbose=False):
        for i in range(len(FACTORS)):
            plt.subplot(2, 3, i + 1)
            fact = []
            alpha = []
            stocks = []
            for stock in self.stocks:
                try:
                    _fact = stock.calcFactor(FACTORS[i], currentTime)
                except:
                    continue
                if not np.isnan(_fact):
                    fact.append(_fact)
                    alpha.append(stock.calcReturn(currentTime, timeFrame))
                    stocks.append(stock)
            plt.scatter(fact, alpha)
            plt.title(
                f"{FACTORS[i]} IC: {self.calcIC(FACTORS[i], currentTime, timeFrame).round(3)}"
            )
            plt.xlabel("Factor Score")
            plt.ylabel("Return")
            if verbose:
                print(
                    pd.DataFrame(
                        {
                            "Stock": stocks,
                            "{}".format(FACTORS[i]): fact,
                            "Return": alpha,
                        }
                    )
                )
        plt.suptitle(f"IC Scatter Plot for {self.name} at {currentTime.date()}")
        plt.tight_layout()
        plt.show()

    # Calculate the ICs for a given factor in a given time
    # 1 day IC, 5 day IC, 10 day IC
    def calcICBulk(self, currentTime, config):
        IC = []
        for timeFrame in config:
            IC += [self.calcIC(factor, currentTime, timeFrame) for factor in FACTORS]
        return IC


# Create a dataframe of ICs for a given stockpile
def createICDataFrame(stockPile, period, config=None):
    ic = []
    nextic = []
    date = []
    delta = (period[1] - period[0]).days
    window = 5

    if config is None:
        config = [1, 5, 10]

    def return20D(currentTime):
        now = stockPile.getIndex(currentTime)
        next = stockPile.getIndex(currentTime + pd.Timedelta(20, unit="d"))
        return (next - now) * 100 / now

    for i in trange(len(period)):
        if i < window:
            continue
        currentTime = period[i]
        IC_append = []
        nextIC_append = stockPile.calcICBulk(
            currentTime + pd.Timedelta(20, unit="d"), [20]
        )
        for j in range(window):
            IC_append += stockPile.calcICBulk(period[i - j], config)
        IC_append += [return20D(currentTime)]
        IC_append += stockPile.getIndicators(currentTime)
        date.append(currentTime)
        ic.append(IC_append)
        nextic.append(nextIC_append)

    Date = pd.DataFrame(date, columns=["Date"])
    col = []
    for _window in range(window):
        for timeFrame in config:
            col += [
                "{}_{}d_{}".format(factor, timeFrame, _window) for factor in FACTORS
            ]
    IC_output = pd.DataFrame(
        ic,
        columns=col
        + ["20 Day Return"]
        + [i.name for i in stockPile.indicators]
        + ["{} YoY".format(i.name) for i in stockPile.indicators]
        + ["Index"],
    )
    nextIC_output = pd.DataFrame(nextic, columns=FACTORS)
    icwithdate = pd.concat([Date, IC_output.iloc[:, 0:6]], axis=1)
    icwithdate.set_index("Date", inplace=True)
    icwithdate = icwithdate.sub(icwithdate.min(axis=1), axis=0).div(
        icwithdate.max(axis=1) - icwithdate.min(axis=1), axis=0
    )
    icwithdate = icwithdate.div(icwithdate.sum(axis=1), axis=0)
    return IC_output, nextIC_output, icwithdate


# %%
stocks = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
]
djia = pd.read_csv(os.path.join(DATA_DIR, "indices", "DJIA.csv"), parse_dates=["Date"])
dow30 = StockPile("DOW30", djia)

for stock in stocks:
    df_price = pd.read_csv(
        os.path.join(DATA_DIR, "stocks", f"{stock}.csv"), parse_dates=["Date"]
    )
    df_stats = pd.read_csv(
        os.path.join(DATA_DIR, "financials", f"{stock}_STAT.csv"), parse_dates=["Date"]
    )
    df_price["Date"].dropna(inplace=True)
    df_stats["Date"].dropna(inplace=True)
    dow30.addStock(Stock(stock, df_price, df_stats))

for indicator in glob.glob(os.path.join(DATA_DIR, "indicators", "*.csv")):
    df_indicator = pd.read_csv(indicator, parse_dates=["DATE"])
    dow30.addIndicator(Indicator(os.path.basename(indicator)[:-4], df_indicator))


# %%
for indicator in dow30.indicators:
    print(indicator.name)

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

FACTORS = [
    "valuation",
    "profitability",
    "growth",
    "quality",
    "liquidity",
    "momentum_and_reversal",
]


class Stock:
    def __init__(self, name, df_price: pd.DataFrame, df_stats: pd.DataFrame):
        self.name = name
        self.df = df_price
        self.dfStatRaw = df_stats.sort_values(by="Date")
        self.dfStat = self.interpolateStats(self.dfStatRaw)

    def __str__(self):
        return self.name

    # Calculate the factor score for a given time
    def calcFactor(self, factor, currentTime):
        dfTime = self.df[self.df["Date"] <= pd.to_datetime(currentTime)].tail(1)
        dfStatTime = self.dfStat[
            self.dfStat["Date"] <= pd.to_datetime(currentTime)
        ].tail(1)
        if factor == "size":
            return dfStatTime.iloc[0]["SIZE"]
        elif factor == "valuation":
            return dfStatTime.iloc[0]["PBR"]
        elif factor == "profitability":
            return dfStatTime.iloc[0]["ROE"]
        elif factor == "growth":
            return dfStatTime.iloc[0]["GROWTH"]
        elif factor == "quality":
            return dfStatTime.iloc[0]["Current Ratio"]
        elif factor == "liquidity":
            return dfTime.iloc[0]["ILLIQ"]
        elif factor == "momentum_and_reversal":
            return self.calcReturn(currentTime, 60)
        else:
            return None

    # Calculate the return for a given time
    def calcReturn(self, currentTime, timeFrame):
        now = (
            self.df[self.df["Date"] <= pd.to_datetime(currentTime)]
            .tail(1)
            .iloc[0]["Close"]
        )
        next = (
            self.df[
                self.df["Date"]
                <= pd.to_datetime(currentTime) + pd.Timedelta(timeFrame, unit="d")
            ]
            .tail(1)
            .iloc[0]["Close"]
        )
        return (next - now) * 100 / now

    # Interpolate the missing values in the stats dataframe
    def interpolateStats(self, df: pd.DataFrame):
        df = df.set_index("Date")
        try:
            df = df.reindex(
                pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq="D",
                )
            )
        except:
            print(self.name)
            print(df.index.min(), df.index.max())
            print(df)
            sys.exit()
        df = df.interpolate(method="cubicspline", axis=0)
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})
        return df


class Indicator:
    def __init__(self, name, df: pd.DataFrame):
        self.name = name
        self.df = df

    def __str__(self):
        return self.name

    def calcIndicator(self, currentTime):
        return (
            self.df[self.df["Date"] <= pd.to_datetime(currentTime)].tail(1).iloc[0, 1]
        )

    def calcYoY(self, currentTime):
        now = self.calcIndicator(currentTime)
        lastYear = self.calcIndicator(currentTime - pd.Timedelta(365, unit="d"))
        return (now - lastYear) * 100 / lastYear

    def calcMoM(self, currentTime):
        now = self.calcIndicator(currentTime)
        lastMonth = self.calcIndicator(currentTime - pd.Timedelta(30, unit="d"))
        return (now - lastMonth) * 100 / lastMonth


class StockPile:
    def __init__(self, name="StockPile", index: pd.DataFrame = None):
        self.name = name
        self.stocks = []
        self.indicators = []
        self.index = index

    def __str__(self):
        return self.name

    def addStock(self, stock: Stock):
        self.stocks.append(stock)

    def addIndicator(self, indicator):
        self.indicators.append(indicator)

    def getIndex(self, currentTime):
        return (
            self.index[self.index["Date"] <= pd.to_datetime(currentTime)]
            .tail(1)
            .iloc[0]["Close"]
        )

    def getIndicators(self, currentTime):
        return (
            [i.calcIndicator(currentTime) for i in self.indicators]
            + [i.calcYoY(currentTime) for i in self.indicators]
            + [i.calcMoM(currentTime) for i in self.indicators]
            + [self.getIndex(currentTime)]
        )

    # Calculate the IC for a given factor in a given time
    def calcIC(self, factor, currentTime, timeFrame=10, verbose=False):
        fact = []
        alpha = []
        for stock in self.stocks:
            try:
                _fact = stock.calcFactor(factor, currentTime)
            except:

                continue
            if not np.isnan(_fact):
                fact.append(_fact)
                alpha.append(stock.calcReturn(currentTime, timeFrame))
        if verbose:
            print(fact)
            print(alpha)
            print(len(fact), len(alpha))

        return np.corrcoef(fact, alpha)[0, 1]

    def showICScatter(self, currentTime, timeFrame=10, verbose=False):
        for i in range(len(FACTORS)):
            plt.subplot(2, 3, i + 1)
            fact = []
            alpha = []
            stocks = []
            for stock in self.stocks:
                try:
                    _fact = stock.calcFactor(FACTORS[i], currentTime)
                except:
                    continue
                if not np.isnan(_fact):
                    fact.append(_fact)
                    alpha.append(stock.calcReturn(currentTime, timeFrame))
                    stocks.append(stock)
            plt.scatter(fact, alpha)
            plt.title(
                f"{FACTORS[i]} IC: {self.calcIC(FACTORS[i], currentTime, timeFrame).round(3)}"
            )
            plt.xlabel("Factor Score")
            plt.ylabel("Return")
            if verbose:
                print(
                    pd.DataFrame(
                        {
                            "Stock": stocks,
                            "{}".format(FACTORS[i]): fact,
                            "Return": alpha,
                        }
                    )
                )
        plt.suptitle(f"IC Scatter Plot for {self.name} at {currentTime.date()}")
        plt.tight_layout()
        plt.show()

    # Calculate the ICs for a given factor in a given time
    # 1 day IC, 5 day IC, 10 day IC
    def calcICBulk(self, currentTime, config):
        IC = []
        for timeFrame in config:
            IC += [self.calcIC(factor, currentTime, timeFrame) for factor in FACTORS]
        return IC


# Create a dataframe of ICs for a given stockpile
def createICDataFrame(stockPile, period, config=None):
    ic = []
    nextic = []
    date = []
    delta = (period[1] - period[0]).days
    window = 5

    if config is None:
        config = [1, 5, 10]

    def return20D(currentTime):
        now = stockPile.getIndex(currentTime)
        next = stockPile.getIndex(currentTime + pd.Timedelta(20, unit="d"))
        return (next - now) * 100 / now

    for i in trange(len(period)):
        if i < window:
            continue
        currentTime = period[i]
        IC_append = []
        nextIC_append = stockPile.calcICBulk(
            currentTime + pd.Timedelta(20, unit="d"), [20]
        )
        for j in range(window):
            IC_append += stockPile.calcICBulk(period[i - j], config)
        IC_append += [return20D(currentTime)]
        IC_append += stockPile.getIndicators(currentTime)
        date.append(currentTime)
        ic.append(IC_append)
        nextic.append(nextIC_append)

    Date = pd.DataFrame(date, columns=["Date"])
    col = []
    for _window in range(window):
        for timeFrame in config:
            col += [
                "{}_{}d_{}".format(factor, timeFrame, _window) for factor in FACTORS
            ]
    IC_output = pd.DataFrame(
        ic,
        columns=col
        + ["20 Day Return"]
        + [i.name for i in stockPile.indicators]
        + ["{} YoY".format(i.name) for i in stockPile.indicators]
        + ["{} MoM".format(i.name) for i in stockPile.indicators]
        + ["Index"],
    )
    nextIC_output = pd.DataFrame(nextic, columns=FACTORS)
    icwithdate = pd.concat([Date, IC_output.iloc[:, 0:6]], axis=1)
    icwithdate.set_index("Date", inplace=True)
    icwithdate = icwithdate.sub(icwithdate.min(axis=1), axis=0).div(
        icwithdate.max(axis=1) - icwithdate.min(axis=1), axis=0
    )
    icwithdate = icwithdate.div(icwithdate.sum(axis=1), axis=0)
    return IC_output, nextIC_output, icwithdate


# %%
# Plot Price and stats in multiple axis
preview = 150

fig = plt.figure(figsize=(15, 12))
ax = plt.subplot(6, 1, 1)
ax.scatter(
    dow30.stocks[0].df["Date"].tail(preview),
    dow30.stocks[0].df["Close"].tail(preview),
    marker=".",
    label="Price",
    color="tab:blue",
)
ax.set_ylabel("Price", color="tab:blue")
ax.tick_params(axis="y", labelcolor="tab:blue")
ax.set_title("Sparcity of Factor Data")
colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
colors.reverse()
for factor in dow30.stocks[0].dfStat.columns[1:]:
    color = colors.pop()
    plt.subplot(6, 1, (6 - len(colors)), sharex=ax)
    plt.plot(
        dow30.stocks[0].dfStat["Date"].tail(preview),
        dow30.stocks[0].dfStat[factor].tail(preview),
        ".",
        label=factor,
        color=color,
    )
    plt.ylabel(factor, color=color)
    plt.tick_params(axis="y", labelcolor=color)


# %%
dow30.calcIC("valuation", "2016-01-08", verbose=True)

# %%
# Scatter IC
for period in TEST_PERIOD[:4]:
    dow30.showICScatter(period + pd.Timedelta(20, unit="d"), 20)

# %%
# Train Dataset
IC_train, nextIC_train, icwithdate = createICDataFrame(dow30, TRAIN_PERIOD, CONFIG)


# %%
IC_train

# %%
print(IC_train.shape)

# %%
# Test Dataset
IC_test, nextIC_test, icwithdate_test = createICDataFrame(dow30, TEST_PERIOD, CONFIG)
print(IC_test.shape, nextIC_test.shape)

# %%
# IC Graph Generation
# icwithdate.plot.area(
# figsize=(20, 5),
# legend=True,
# colormap="tab10",
# title="IC Weights over training period",
# )
# plt.savefig(dpi=300, fname="icweights.png")


# %%
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

xgbModel = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2,
        gamma=0,
        subsample=0.2,
        colsample_bylevel=0.6,
        colsample_bytree=0.2,
        reg_lambda=20,
        tree_method="hist",
    )
)

# %%
# Model Training
print(IC_train.shape, nextIC_train.shape)
xgbModel.fit(IC_train, nextIC_train)
print(xgbModel.score(IC_train, nextIC_train))


# %%
plt.figure(figsize=(60, 8))
IC_predict_train = pd.DataFrame(xgbModel.predict(IC_train), columns=FACTORS)
for i in range(len(FACTORS)):
    plt.subplot(2, 3, i + 1)
    plt.title(FACTORS[i], fontsize=16)
    plt.plot(IC_predict_train.iloc[:, i], label="prediction")
    plt.plot(nextIC_train.iloc[:, i], label="real")
    plt.legend()

plt.savefig(dpi=300, fname="prediction.png")

# %%
print(xgbModel.score(IC_test, nextIC_test))


# %%
# Model Testing
print(xgbModel.score(IC_test, nextIC_test))
plt.figure(figsize=(18, 8))
plt.suptitle("IC Weights over testing period", fontsize=20)
IC_predict = pd.DataFrame(xgbModel.predict(IC_test), columns=FACTORS)
for i in range(len(FACTORS)):
    plt.subplot(2, 3, i + 1)
    plt.title(FACTORS[i], fontsize=16)
    plt.plot(IC_predict.iloc[:, i], label="prediction")
    plt.plot(nextIC_test.iloc[:, i], label="real")
    plt.legend()

plt.savefig(dpi=300, fname="prediction.png")
