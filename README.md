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
   
## Datasets
1. **COVID-19 Data**
   - This data is provided in [Public data portal](https://www.data.go.kr/data/15098776/openapi.do) by MoHW. 
   - This data has the number of COVID-19 cases by city.
   - Columns
      - **deathCnt** : Number of people who died on day <br>
      - **defCnt** : Cumulative number of cases on day <br>
      - **gubun** : City name in Korean <br>
      - **gubunEn** : City name in English <br>
      - **incDec** : Number of cases on day <br>
      - **isolClearCnt** : Number of people who recovered on day <br>
      - **isollngCnt** : Number of people in quarantine on day <br>
      - **localOccCnt** : Number of cases in Korea on day <br>
      - **overFlowCnt** : Number of cases arriving from abroad on day <br>
      - **qurRate** : Cases per 100,000 population on day <br>
      - **stdDay** : Date (2020.01 ~ 2023.05) <br>
      
3. **Variants**
   - This data is provided in [Covariants](https://covariants.org/).
   - This data has the stricgency index that quantifies the policy response on day.
   - We use Alpha, Delta, Omicron(22D, 23A).

4. **Stricgency**
   - This data is provided in [Our World in Data](https://ourworldindata.org/covid-stringency-index) by 'OxCGRT'.
   - If you want to know how it was calculated, you can go [here](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md)
## Processing
1. Difference
      > **ACn** : Accumulation number of confirmed cases on day **n** <br>
      > **DCn** : Daily number of confirmed cases on day **n** (Difference of **ACn**) <br>
      > **DDCn** : Difference of **DCn** <br>
      > **DDDCn** : Difference of **DDCn** <br>   
