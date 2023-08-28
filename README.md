# Forex_Prediction_Using_Deep_Learning
<p align='justify'>
In this repo we establish and compare three different AI models: <strong> LSTM </strong>, <strong> LGBM </strong>, <strong> INFORMER </strong> and two 
different data modelling techniques: <strong> Forex Modelling </strong> and <strong> Forex Return Modelling </strong>. The dataset we used comprised of the hourly forex data of four key 
exchange rates: <strong> GBP/USD, EUR/GBP, EUR/USD, and XAU/USD </strong>. Our experimental results highlight the stochastic nature of markets which poses a 
challenge to state of the art models in achieving reliable predictions.
</p>
<p align='justify'>
Data preprocessing played a pivotal role in empowering the models by removing large swings and 
allowing them capture the underlying patterns and relationships present in the data. Feature 
Engineering, on the other hand, had a negative impact on the performance of the models as the best 
results were achieved with just the exchange rates and their returns as inputs.
</p>
<p align="center" width="75%">
    <img width="75%" src="https://github.com/Cheeetah97/Forex_Prediction_Using_Deep_Learning/assets/62606459/ff9db2ec-1071-4c15-a4f8-db1274c1557e"> 
</p>
<p align='justify'>
Of the three AI models, LightGBM produced the best results beating both state of the art deep learning 
models for time series forecasting. This finding is significant because it addresses the misconception of 
“Complex Models are always better”. While complex models are beneficial for certain tasks they are 
not necessarily a general solution especially in financial forecasting where overfitting on noise can be 
a huge problem. A simpler model can therefore perform better in such scenarios.
</p>
<p align='justify'>
In Table 2, the MSEs are compared to the Mean Value of Return of the test portions. The mean value 
of return, in the context of financial forecasting, is considered as a naïve forecast.
</p>
<p align="center" width="75%">
    <img width="75%" src="https://github.com/Cheeetah97/Forex_Prediction_Using_Deep_Learning/assets/62606459/83254293-d0f9-41f0-811e-aa4487f42e35"> 
</p>

<p align='justify'>
Similarly in Table 3, we compared the Directional Accuracy Scores of each Model.
</p>
<p align="center" width="75%">
    <img width="75%" src="https://github.com/Cheeetah97/Forex_Prediction_Using_Deep_Learning/assets/62606459/5456db90-070b-499a-8827-ae7f08b207fd"> 
</p>

<p align='justify'>
We further demonstrated the performance of the models by evaluating them in a simulated trading environment. We used the predictions of the best performing 
model i.e., LGBM and created a trading strategy. The trading strategy was implemented on each exchange rate for each one of the five test portions. It used 
two indicators: one for actual rate increment and the other for predicted rate increment. Based on these indicators, the buy and sell actions were triggered. 
When the model predicted a rise in the exchange rate and the actual rate confirmed it, it resulted in a profit. Conversely, when the model predicted an 
increase that didn’t materialize, a loss was incurred. The cumulative net profit was then computed over time, reflecting the overall success of the trading approach.
</p>
<p align='justify'>
We began by investing a base amount of 20,000 units of the base currency on an hourly basis throughout 
the specified test portions. If we consider a minimum transaction cost of 1 unit of the base currency per trade, the cumulative 
profit graphs looked like as depicted in Fig.9.
</p>
<p align="center" width="75%">
    <img width="75%" src="https://github.com/Cheeetah97/Forex_Prediction_Using_Deep_Learning/assets/62606459/078a0233-3a98-40e2-9746-429f796cf6fb"> 
</p>

<p align='justify'>
This repo also contains the implementation of Additive Attention layer and a Custom Loss Function.
</p>
