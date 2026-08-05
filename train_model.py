import pandas as pd 

df=pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['Churn'].value_counts())