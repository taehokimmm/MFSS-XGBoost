import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # Calculate the indicator value for a given time
    def calcIndicator(self, currentTime):
        return (
            self.df[self.df["DATE"] <= pd.to_datetime(currentTime)].tail(1).iloc[0, 1]
        )

    # Calculate the Year over Year change for a given time
    def calcYoY(self, currentTime):
        now = self.calcIndicator(currentTime)
        lastYear = self.calcIndicator(currentTime - pd.Timedelta(365, unit="d"))
        return (now - lastYear) * 100 / lastYear

    # Calculate the Month over Month change for a given time
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

    # Add a stock to the stockpile
    def addStock(self, stock: Stock):
        self.stocks.append(stock)

    # Add an indicator to the stockpile
    def addIndicator(self, indicator):
        self.indicators.append(indicator)

    # Get the index value for a given time
    def getIndex(self, currentTime):
        # Just in case the date is a holiday
        return (
            self.index[self.index["Date"] <= pd.to_datetime(currentTime)]
            .tail(1)
            .iloc[0]["Close"]
        )

    # Get all the indicator scores for a given time
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
                # just in case the stock doesn't have the factor
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


# Create a dataframe of ICs for a given stockpile, used as a dataloader in training and testing
def createICDataFrame(stockPile, period, config=None):
    ic = []
    nextic = []
    date = []
    delta = (period[1] - period[0]).days
    window = 5 # input window size

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
