# Digital Signal Processing

You are at the heart of the algorithms used by the [TradingInPython](https://www.trading-et-data-analyses.com/p/plateforme-de-trading-technique.html) platform.

This is where you can find the algorithms that make up the trading strategies. It's the open source part of the platform.

You can take a look at the algorithms and understand in depth the indicators used in the trading strategies.

## Open source files

You have two sets of files:

- scripts in python file like **.py**

- those scripts compiled in **.c** **.pyd**

The Plateform uses the compiled files to run the algorithms if they are present, if not it uses the compiled files.

This is **not recommended** but you could very well suppress the compiled files and run the platform with the python scripts files. So you will be able **to modified them**.

If you like this work I highly recommend you to [subscribe](https://www.trading-et-data-analyses.com/p/abonnement.html).

## Indicators

The indicators are the algorithms that will allow you to make decisions on the market.

Here are the indicators used in the trading strategies:

- [Digital signal processing/indicators.py](indicators.py)

## Ichimoku Kinko Hyo

The Ichimoku Kinko Hyo strategy is a very popular strategy in trading. Here we have modernized it by using **deep learning** and Keras neural networks.

You can find the algorithm that implements the Ichimoku Kinko Hyo strategy here:

- [Ichimoku Kinko Hyo prediction alorithms.py](ichimoku_kinko_hyo.py)

## Other strategies

You will discover many other strategies by downloading and running the platform **TradingInPyhton**.
