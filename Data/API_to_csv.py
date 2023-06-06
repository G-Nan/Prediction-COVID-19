import xmltodict
import requests
import pandas as pd

def load_API(url_city, url_base, url_serviceKey):
    df_list = []
    
    for page_num in range(1, 4):
        url = url_base + '?serviceKey=' + url_serviceKey + '&pageNo=' + str(page_num) + '&gubun=' + url_city
        req = requests.get(url).content
        xmlObject = xmltodict.parse(req)
        df_sub = pd.DataFrame(xmlObject['response']['body']['items']['item'])
        df_list.append(df_sub)
        
    df = pd.concat(df_list)
    df.drop('gubunCn', axis = 1, inplace = True)
    df.sort_values(by = 'stdDay', inplace = True)   
    df.reset_index(drop = True, inplace = True)    
    
    return df

#url_base = "http://apis.data.go.kr/1352000/ODMS_COVID_04/callCovid04Api"
#url_serviceKey = "vh1TU%2B6gkd3lndqMovQxTWu1mREkKj1aEAZyioG4LKCqrp3HiBs30H56eBWWna7m6IlqjyDZHvmJNtII6G27Qw%3D%3D"
#cities = ['서울', '부산', '대구', '인천', '광주', '대전', 
#          '울산', '세종', '경기', '강원', '충북', '충남', 
#          '전북', '전남', '경북', '경남', '제주']


def api_to_csv(cities, url_base, url_serviceKey):
    for city in cities:
        try:
            df = load_API(city, url_base, url_serviceKey)
        except:
            print(city + '- Load Error')
        
        try:
            df.to_csv(f"Data/RawData/{city}_Raw.csv", encoding = 'cp949', index = False)
            print(city + ' - OK')
        except:
            print(city + ' - Save Error')