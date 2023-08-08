# Prediction-COVID-19 (2023-02-22 ~ 2023-06-30)
Predicting the number of COVID-19 infections.

## Description
- We predicted the number of COVID-19 infections in each of 17 cities in South Korea. 
- We predicted the coronavirus for each variant separately and analyze them. (Alpha/Delta, Delta/Omicron, Omicron22D/Omicron23A)
- We predicted for the next 7, 14 and 21 days. 
- We used differential data for more accurate prediction. 
- We applied SIR model to the network for more accurate predictions. 
- We used DL like RNN, LSTM, GRU, Bidirection, seq2seq and PINN. 
- We wanted to compare the accuracy of our methods using RMSE and MAPE and find the best one. 
- We finally evaluate the number of COVID-19 cases against South Korea's stricgency index. 

## Methods
1. **Many-to-One** 
   - Predicting the day ahead and then use that prediction again to predict the next day's data.
   - For example, if you want to predict 7 days, you repeat the action of predicting the next day 7 times to predict 7 days.
   
2. **Many-to-Many**
   - Predicting all target days at a time.
   - For example, if you want to predict 7 days, you predict all 7 days at once using seq2seq model with RNN, LSTM, GRU.

3. **SIR**
   - Utilizing the SIR model to first predict the infection rate(**alpha**), then calculate the number of cases.
   - In this case, we didn't use the number of cases data(**COVID-19**), but the infection rate data(**SIR**).
   - For example, if you want to predict 7 days, you would first predict the infection rate for 7 days and then calculate it as the numer of cases

4. **PINN**
   - Solving differential equations in SIR model using artificial neural networks.
   - If you want to know more information about PINN, we recommend reading the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)


## Datasets
1. **COVID-19 Data**
   - This data is provided in [Public data portal](https://www.data.go.kr/data/15098776/openapi.do) by MoHW. 
   - This data has the number of COVID-19 cases by city.
   - Columns
      - **deathCnt** : Number of people who died on day.
      - **defCnt** : Cumulative number of cases on day.
      - **gubun** : City name in Korean.
      - **gubunEn** : City name in English.
      - **incDec** : Number of cases on day.
      - **isolClearCnt** : Number of people who recovered on day.
      - **isollngCnt** : Number of people in quarantine on day.
      - **localOccCnt** : Number of cases in Korea on day.
      - **overFlowCnt** : Number of cases arriving from abroad on day.
      - **qurRate** : Cases per 100,000 population on day.
      - **stdDay** : Date (2020.01 ~ 2023.05).
      
3. **Variants**
   - This data is provided in [Covariants](https://covariants.org/).
   - This data has the stricgency index that quantifies the policy response on day.
   - We use Alpha, Delta, Omicron(22D, 23A).

4. **Stricgency**
   - This data is provided in [Our World in Data](https://ourworldindata.org/covid-stringency-index) by 'OxCGRT'.
   - If you want to know how it was calculated, you can go [here](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md)
     
## Data Processing
1. **Difference**
   - We used differential data for more accurate prediction.
   - Columns
      - **ACn** : Accumulation number of confirmed cases on day **n**.
      - **DCn** : Daily number of confirmed cases on day **n** (Difference of **ACn**).
      - **DDCn** : Difference of **DCn**.
      - **DDDCn** : Difference of **DDCn**.

2. **Variants**
   - We devided the COVID-19 data by variant.
   - Example
     > |Date|Daily cases|Delta variant rate|Omicron variant rate|
     > |:--:|:--:|:--:|:--:|
     > |2022.01.01|100|0.2|0.8|
     In this case, the number of people infected with the Delta variant is 20, Omicron is 80.

3. **SIR**
   - We applied SIR model to the network for more accurate predictions. <br>
   - Columns
      - **Susceptible** : People who are susceptible to infection.
      - **Infected** : People who are infected.
      - **Recovered** : People who are recovered.
      - **Deceased** : People who are deceased.
      - **Alpha** : The propability of a contact resulting in infection. (Force of infection)
      - **Beta** : Recovery rate
      - **Gamma** : Death rate
    - Calculation method
      - We assumed that once infected, a person would either be cured or die after 14 days. (If omicron, 7 days)
      - **Infected**
         > $$I_{t} = \sum_{i=t-13}^{t}{I_i}$$
         
      - **Deceased**
         > $$D_{t} = \sum_{i=1}^{t}{D_i}$$
         
      - **Recovered**
         > $$R_{t} = I_{t-14}$$

      - **Susceptible**
         > $$S_{t} = N - I_{t} - D_{t} - R_{t}$$

      - **Alpha**
         > $$\alpha_{t} = \frac{-1 \times N \times (S_{t+1}-S_{t})}{S_{t} \times I_{t}}$$

      - **Beta**
         > $$\beta_{t} = \frac{R_{t+1}-R_{t}}{I_{t}}$$

      - **Gamma**
         > $$\gamma_{t} = \frac{D_{t+1}-D_{t}}{I_{t}}$$

## Prediction
1. **All piriods**
   - We predicted the number of COVID-19 cases for all time periods in 17 cities.
   
2. **Alpha/Delta**
   - We predicted  the number of COVID-19 cases for the Alpha and Delta variants in 17 cities and analyzed it. 

3. **Omicron 22D/23A**
   - We predicted the number of COVID-19 cases for the Omicron variants 22D and 23A in 17 cities and analyzed it. 

4. **The moment a policy changes**
   - We predicted the number of COVID-19 cases at the moment of the policy change and analyzed it to evaluate the policy change.
   - We selected four policy change(two up, two down).

## Hyperparameter
  - We utilized GridSearch to find the optimal parameters.
  - The candidates for the parameters we set are
      - Learning Rate : 1e-3, 1e-4, 1e-5 <br>
      - Batch Size : 32, 64, 128 <br>
      - Num_Layers : 1, 2, 4, 8 <br>
      - Hidden_Size : 4, 8, 16, 32 <br>
      - Patience(Early Stopping) : 20, 50, 100 <br>
      - Dropout : 0.25, 0.5 <br>
      - Loss Function : MSELoss, criterion2, criterion3 <br>
        > criterion2 <br>
           $$\sum_{i=0}^{7\ or\ 14}{i \times |A_{i} - P_{i}|}$$ <br>
        > criterion3 <br>
          $$\sum_{i=0}^{7\ or\ 14}{i^2 \times (A_{i} - P_{i})^2}$$
  - We trained all the candidates for each data and selected the one with the lowest MAPE.
  - There are 4 variants and 4 time periods of data for each of the 17 cities, so a total of 153 data points were trained. 
      - 17 cities : Gangwon, Gyeonggi, Gyeongnam, Gyeongbuk, Gwangju, Daegu, Daejeon, Busan, Seoul, <br>　　　　 Sejong, Ulsan, Incheon, Jeonnam, Jeonbuk, Jeju, Chungnam, Chungbuk
      - 4 variants : Alpha, Delta, Omicron 22D, Omicron 23A
      - 4 time period : Two periods of increased quarantine policy strength and two periods of decreased strength.

## About Us
|<img src="https://github.com/chdaewon.png" width="80">|<img src="https://github.com/G-Nan.png" width="80">|<img src="https://github.com/ddanggu.png" width="80">|<img src="https://github.com/Ahnjihyeok.png" width="80">|
|:---:|:---:|:---:|:---:|
|[chdaewon](https://github.com/chdaewon)|[G-Nan](https://github.com/G-Nan)|[ddanggu](https://github.com/ddanggu)|[akrehd2](https://github.com/Ahnjihyeok)|
|PM, PINN|RNN, Statistics|Data Management, PINN|EDA, Visualization|



















