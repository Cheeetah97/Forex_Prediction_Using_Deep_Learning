# Forex_Prediction_Using_Deep_Learning

In this repo we establish and compare three different AI models: **LSTM**, **LGBM**, **Informer** and two 
different data modelling techniques: **Forex Modelling** and **Forex Return Modelling**. The dataset we used comprised of the hourly forex data of four key 
exchange rates: **GBP/USD, EUR/GBP, EUR/USD, and XAU/USD**. Our experimental results highlight the stochastic nature of markets which poses a 
challenge to state of the art models in achieving reliable predictions. 
Data preprocessing played a pivotal role in empowering the models by removing large swings and 
allowing them capture the underlying patterns and relationships present in the data. Feature 
Engineering, on the other hand, had a negative impact on the performance of the models as the best 
results were achieved with just the exchange rates and their returns as inputs.
Of the three AI models, LightGBM produced the best results beating both state of the art deep learning 
models for time series forecasting. This finding is significant because it addresses the misconception of 
“Complex Models are always better”. While complex models are beneficial for certain tasks they are 
not necessarily a general solution especially in financial forecasting where overfitting on noise can be 
a huge problem. A simpler model can therefore perform better in such scenarios.
We further demonstrated the performance of the models by evaluating them both in terms of their 
predictive power and their suitability in trading.
This repo also contains the implementation of Additive Attention layer and a Custom Loss Function.
