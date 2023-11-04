import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import json
token=st.secrets['token']
def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://adb-1932438962374148.8.azuredatabricks.net/serving-endpoints/xgboost/invocations'
    headers = {'Authorization': f'Bearer {token}','Content-Type': 'application/json'}
    ds_dict = {"dataframe_split": dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

st.title('Databricks App')
age=st.number_input('Enter Your age',min_value=18,value='min')
work_class=st.selectbox('Seelect workclass',['Non-Private', 'Private'])
fnlwgt=st.number_input('Enetr fnlwgt')
education=st.selectbox('Select your education',['Bachelors', 'HS-grad', 'other-education', 'Masters', 'Some-college'])
edu_num=st.selectbox('Enter education number',[13, 9, 7, 14, 5, 10, 12, 11, 4, 16, 15, 3, 6, 2, 1, 8])
marital_status=st.selectbox('Select marital status',['Never-married', 'Married-civ-spouse', 'Divorced', 'others'])
occupation=st.selectbox('Select occupation',['Adm-clerical','Exec-managerial','Other-service','Prof-specialty', 'Sales','Craft-repair'])
relation=st.selectbox('Select relationship',['Not-in-family', 'Husband', 'others', 'Own-child', 'Unmarried'])
race=st.selectbox('Select race',['White', 'Non-White'])
sex=st.selectbox('Select sex',['Male', 'Female'])
hrs_week=st.number_input('Enter hours per week')
data={
    "age":age,
   "workclass":work_class,
   "fnlwgt":fnlwgt,
   "education":education,
   "education-num":edu_num,
   "marital-status":marital_status,
   "occupation":occupation,
   "relationship":relation,
   "race":race,
   "sex":sex,
   "hours-per-week":hrs_week
    }
df=pd.DataFrame(data,index=[0])
ordinal=OrdinalEncoder(categories=[['other-education', 'Some-college','HS-grad','Bachelors','Masters']])
df['education']=ordinal.fit_transform(df[['education']])
if st.button('Predict'):
    response=score_model(df)
    if response['predictions'][0]==0:
        st.write('<$50k')
    else:
        st.write('>$50k')


