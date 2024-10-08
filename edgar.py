from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import ratelimit
from finqual import finqual as fq



def getCompanyData(ticker, start, end):
    income = fq.Ticker(ticker).income(start.year, end.year, quarter=True).transpose()
    balance = fq.Ticker(ticker).balance(start.year, end.year, quarter=True).transpose()
    cashflow = fq.Ticker(ticker).cashflow(start.year, end.year, quarter=True).transpose()

    companydf = pd.merge(income, balance, left_index=True, right_index=True)
    companydf = pd.merge(companydf, cashflow, left_index=True, right_index=True)
    companydf.set_index(pd.PeriodIndex(companydf.index, freq='Q'), inplace=True)
    return companydf

if __name__ == '__main__':
    end = datetime.now()
    start = (end - relativedelta(years=10))
    data = getCompanyData("AAPL", start, end)
    json_result = data.to_json(orient='columns', index=True)
    print(json_result)
