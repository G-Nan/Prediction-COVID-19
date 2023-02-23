# Prediction-COVID-19
Predicting the number of COVID-19 infections

## Description
- We predict the number of COVID-19 infections in each of 17 cities in South Korea. <br>
- We predict for the next 7 days. <br>
- We used DL like RNN, LSTM and Pytorch Forecasting. <br>
- We wanted to compare the accuracy of our methods and find the best one. <br>

## Methods
1. Many-to-One 
   -  Predict the day ahead and then use that prediction again to predict the next day's data.
   -  Repeat these actions 7 times to predict a total of 7 days of values.
   
2. Many-to-Many
   - Predict 7 days of data at a time.
   
3. Pytorch Forecasting
   - Predict 7 days using Pytorch Forecasting
   
## Data
- Date : 2020.07 ~
- ACn : 
- DCn : 
- DDCn : 
- DDDCn : 

## Model
- Batch size = 100

