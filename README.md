This is Fork branch from [Microsoft Qlib](https://github.com/microsoft/qlib)

I changed [this folder](https://github.com/HyeongminMoon/qlib_predict_daily-stock_prices/tree/main/examples/benchmarks/TRA), but It's not organized.
I recommand just skip this repository and go to official, Please contact me if you have any interesting

What I did
----------
* I used Qlib's banchmark, TRA models to predict USA(Nasdaq100, SP500) daily stock prices

What I changed&enhanced
---------------
* Preparing custom US stock price Dataset(2008 - 2021 (recent))
* Making Pipeline(auto crawling->normalize->predict->backtest&make decision->auto stock control)
* Custom Strategy & Backtest
