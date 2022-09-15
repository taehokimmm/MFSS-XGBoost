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
    def __init__(self, name, df_price: pd.DataFrame, df_stats: pd.DataFrame = None):
        self.name = name
        self.df = df_price
        self.dfStat = df_stats

    def __str__(self):
        return self.name

    def calcFactor(self, factor, currentTime):
        dfTime = self.df[self.df["Date"] <= pd.to_datetime(currentTime)].tail(1)
        dfStatTime = self.dfStat[self.dfStat.Date <= pd.to_datetime(currentTime)].tail(
            1
        )
        if factor == "size":
            return dfStatTime["SIZE"]
        elif factor == "valuation":
            return dfStatTime["PBR"].mean()
        elif factor == "profitability":
            return dfStatTime["ROE"].mean()
        elif factor == "growth":
            return dfStatTime["GROWTH"].mean()
        elif factor == "quality":
            return dfStatTime["Current Ratio"].mean()
        elif factor == "liquidity":
            return dfTime["ILLIQ"].mean()
        elif factor == "momentum_and_reversal":
            return (
                self.df[self.df["Date"] <= pd.to_datetime(currentTime)]
                .tail(1)
                .iloc[0]["Close"]
                - self.df[
                    self.df["Date"]
                    <= pd.to_datetime(currentTime) - pd.Timedelta(60, unit="d")
                ]
                .tail(1)
                .iloc[0]["Close"]
            )
        else:
            return None

    def calcReturn(self, currentTime, timeFrame):
        return (
            self.df[self.df["Date"] <= pd.to_datetime(currentTime)]
            .tail(timeFrame)
            .iloc[0]["Close"]
            - self.df[
                self.df["Date"]
                <= pd.to_datetime(currentTime) - pd.Timedelta(timeFrame, unit="d")
            ]
            .tail(1)
            .iloc[0]["Close"]
        )


class StockPile:
    def __init__(self, name="StockPile", index=None):
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

    def calcIC(self, factor, currentTime, timeFrame=1):
        factor = [stock.calcFactor(factor, currentTime) for stock in self.stocks]
        alpha = [stock.calcReturn(currentTime, timeFrame) for stock in self.stocks]
        return np.corrcoef(factor, alpha)[0][1] * 10

    def calcICBulk(self, currentTime):
        return (
            [self.calcIC(factor, currentTime, 1) for factor in FACTORS]
            + [self.calcIC(factor, currentTime, 5) for factor in FACTORS]
            + [self.calcIC(factor, currentTime, 10) for factor in FACTORS]
        )


def createICDataFrame(stockPile, period):
    ic = []
    nextic = []
    date = []

    for i in trange(len(period)):
        IC_append = stockPile.calcICBulk(period[i]) + [stockPile.getIndex(period[i])]
        nextIC_append = (
            [stockPile.calcIC(factor, period[i + 1]) for factor in FACTORS]
            if i < len(period) - 1
            else [
                stockPile.calcIC(factor, period[i] + pd.Timedelta(days=7))
                for factor in FACTORS
            ]
        )
        if not np.isnan(IC_append).any() and not np.isnan(nextIC_append).any():
            date.append(period[i])
            ic.append(IC_append)
            nextic.append(nextIC_append)

    Date = pd.DataFrame(date, columns=["Date"])
    IC_output = pd.DataFrame(
        ic,
        columns=[factor + "1" for factor in FACTORS]
        + [factor + "5" for factor in FACTORS]
        + [factor + "10" for factor in FACTORS]
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
