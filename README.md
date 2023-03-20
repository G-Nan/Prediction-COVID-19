# Prediction-COVID-19 (2023-02-22 ~ ing)
Predicting the number of COVID-19 infections.

## Description
- We predict the number of COVID-19 infections in each of 17 cities in South Korea. <br>
- We predict for the next 7 days. <br>
- We used DL like RNN, LSTM and Pytorch Forecasting. <br>
- We wanted to compare the accuracy of our methods and find the best one. <br>

## Methods
1. Many-to-One 
   - Predict the day ahead and then use that prediction again to predict the next day's data.
   - Repeat these actions 7 times to predict a total of 7 days of values.
   
2. Many-to-Many
   - Predict 7 days of data at a time.
   - Using seq2seq model with RNN, LSTM, GRU.
   - For the visualization, we only used the first prediction. (If you have a better idea, please let me know.)
   
3. Pytorch Forecasting
   - Predict 7 days using Pytorch Forecasting
   
## Data
- Date : 2020.07 ~
- ACn : Accumulation number of confirmed cases on day **n**
- DCn : Daily number of confirmed cases on day **n** (Difference of **ACn**)
- DDCn : Difference of **DCn**
- DDDCn : Difference of **DDCn**

## Model
 Many-to-One
> 1. RNN
>     - Hyper Parameter
>        - Learning Rates : 1e-3
>        - 
