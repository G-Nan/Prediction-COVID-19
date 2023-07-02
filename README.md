# Prediction-COVID-19 (2023-02-22 ~ 2023-06-30)
Predicting the number of COVID-19 infections.

## Description
- We predict the number of COVID-19 infections in each of 17 cities in South Korea. <br>
- We predict the coronavirus for each variant separately and analyze them. (Alpha/Delta, Delta/Omicron, Omicron22D/Omicron23A) <br>
- We predict for the next 7, 14 and 21 days. <br>
- We use differential data for more accurate prediction. <br>
- We apply SIR model to the network for more accurate predictions. <br>
- We used DL like RNN, LSTM, GRU, Bidirection, seq2seq and PINN. <br>
- We wanted to compare the accuracy of our methods using RMSE and MAPE and find the best one. <br>
- We finally evaluate the number of COVID-19 cases against South Korea's stricgency index. <br>

## Methods
1. **Many-to-One** 
   - Predicting the day ahead and then use that prediction again to predict the next day's data.
   - For example, if we want to predict 7 days, we repeat the action of predicting the next day 7 times to predict 7 days.
   
2. **Many-to-Many**
   - Predicting all target days at a time.
   - For example, if we want to predict 7 days, we predict all 7 days at once using seq2seq model with RNN, LSTM, GRU.
   
## Data
- **Date** : 2020.07 ~
- **ACn** : Accumulation number of confirmed cases on day **n**
- **DCn** : Daily number of confirmed cases on day **n** (Difference of **ACn**)
- **DDCn** : Difference of **DCn**
- **DDDCn** : Difference of **DDCn**
