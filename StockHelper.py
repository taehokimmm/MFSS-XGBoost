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
        self.dfStatRaw = df_stats.sort_values(by="Date").dropna()
        self.dfStat = self.interpolateStats(self.dfStatRaw)

    def __str__(self):
        return self.name

    # Calculate the factor score for a given time
    def calcFactor(self, factor, currentTime, timeFrame):
        dfTime = self.df[self.df["Date"] <= pd.to_datetime(currentTime)].tail(
            timeFrame + 1
        )
        dfStatTime = self.dfStat[
            self.dfStat["Date"] <= pd.to_datetime(currentTime)
        ].tail(timeFrame + 1)
        if factor == "size":
            return dfStatTime["SIZE"].to_numpy()
        elif factor == "valuation":
            return dfStatTime["PBR"].to_numpy()
        elif factor == "profitability":
            return dfStatTime["ROE"].to_numpy()
        elif factor == "growth":
            return dfStatTime["GROWTH"].to_numpy()
        elif factor == "quality":
            return dfStatTime["Current Ratio"].to_numpy()
        elif factor == "liquidity":
            return dfTime["ILLIQ"]
        elif factor == "momentum_and_reversal":
            return np.array(
                [
                    self.df[self.df["Date"] <= time].tail(1).iloc[0]["Close"]
                    - self.df[self.df["Date"] <= time - pd.Timedelta(60, unit="d")]
                    .tail(1)
                    .iloc[0]["Close"]
                    for time in pd.date_range(
                        start=pd.to_datetime(currentTime)
                        - pd.Timedelta(timeFrame, unit="d"),
                        end=pd.to_datetime(currentTime),
                        freq="D",
                    )
                ]
            )
        else:
            return None

    # Calculate the return for a given time
    def calcReturn(self, currentTime, timeFrame):
        return np.array(
            [
                self.df[self.df["Date"] <= time].tail(1).iloc[0]["Close"]
                for time in pd.date_range(
                    start=pd.to_datetime(currentTime)
                    - pd.Timedelta(timeFrame, unit="d"),
                    end=pd.to_datetime(currentTime),
                    freq="D",
                )
            ]
        )

    # Interpolate the missing values in the stats dataframe
    def interpolateStats(self, df: pd.DataFrame):
        df = df.set_index("Date")
        df = df.reindex(
            pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq="D",
            )
        )
        df = df.interpolate(method="cubicspline", axis=0)
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})
        return df


class StockPile:
    def __init__(self, name="StockPile", index: pd.DataFrame = None):
        self.name = name
        self.stocks = []
        self.index = index

    def __str__(self):
        return self.name

    def addStock(self, stock: Stock):
        self.stocks.append(stock)

    def getIndex(self, currentTime):
        return (
            self.index[self.index["Date"] <= pd.to_datetime(currentTime)]
            .tail(1)
            .iloc[0]["Close"]
        )

    # Calculate the IC for a given factor in a given time
    def calcIC(self, factor, currentTime, slide=False, timeFrame=10, kernel=5):
        IC = []
        if slide:
            stride = 1
        else:
            stride = kernel

        for stock in self.stocks:
            fact = avgPool(
                stock.calcFactor(factor, currentTime, timeFrame + kernel),
                kernel,
                stride,
            )
            alpha = avgPool(
                stock.calcReturn(currentTime, timeFrame + kernel), kernel, stride
            )
            IC.append(np.corrcoef(fact, alpha)[0, 1])
        return np.mean(IC)

    # Calculate the ICs for a given factor in a given time
    # 1 day IC, 5 day IC, 10 day IC
    def calcICBulk(self, currentTime, config=None):
        IC = []
        if config is None:
            config = [[True, 1, 5], [True, 5, 5], [True, 10, 5]]
        for _slide, _timeFrame, _kernel in config:
            IC += [
                self.calcIC(
                    factor,
                    currentTime,
                    slide=_slide,
                    timeFrame=_timeFrame,
                    kernel=_kernel,
                )
                for factor in FACTORS
            ]
        return IC


# Create a dataframe of ICs for a given stockpile
def createICDataFrame(stockPile, period, config=None):
    ic = []
    nextic = []
    date = []

    if config is None:
        config = [[True, 1, 5], [True, 5, 5], [True, 10, 5]]

    for i in trange(len(period)):
        IC_append = stockPile.calcICBulk(period[i], config) + [
            stockPile.getIndex(period[i])
            - stockPile.getIndex(period[i] - pd.Timedelta(20, unit="d"))
        ]
        nextIC_append = (
            [
                stockPile.calcIC(
                    factor, period[i + 1], slide=True, timeFrame=10, kernel=5
                )
                for factor in FACTORS
            ]
            if i < len(period) - 1
            else [
                stockPile.calcIC(
                    factor,
                    period[i] + pd.Timedelta(days=10),
                    slide=True,
                    timeFrame=10,
                    kernel=5,
                )
                for factor in FACTORS
            ]
        )
        if not np.isnan(IC_append).any() and not np.isnan(nextIC_append).any():
            date.append(period[i])
            ic.append(IC_append)
            nextic.append(nextIC_append)

    Date = pd.DataFrame(date, columns=["Date"])
    col = []
    for _slide, _timeFrame, _kernel in config:
        col += [
            "{}_{}d_{}k{}".format(factor, _timeFrame, _kernel, "_slide" if _slide else "")
            for factor in FACTORS
        ]
    IC_output = pd.DataFrame(
        ic,
        columns=col + ["20 Day Return"],
    )
    nextIC_output = pd.DataFrame(nextic, columns=FACTORS)
    icwithdate = pd.concat([Date, IC_output.iloc[:, 0:6]], axis=1)
    icwithdate.set_index("Date", inplace=True)
    icwithdate = icwithdate.sub(icwithdate.min(axis=1), axis=0).div(
        icwithdate.max(axis=1) - icwithdate.min(axis=1), axis=0
    )
    icwithdate = icwithdate.div(icwithdate.sum(axis=1), axis=0)
    return IC_output, nextIC_output, icwithdate


# Average pool a given array
def avgPool(arr: np.ndarray, kernel=1, stride=1):
    arr = np.flip(arr)
    return np.flip(
        np.array(
            [np.mean(arr[i : i + kernel]) for i in range(0, len(arr) - kernel, stride)]
        )
    )
