import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder,StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib as jb
df=pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges']=df['TotalCharges'].replace(' ',np.nan)
df['TotalCharges']=df['TotalCharges'].fillna(0)
df['TotalCharges']=df['TotalCharges'].astype(float)

x=df.drop('Churn',axis=1)
y=df['Churn']
num_col=x.select_dtypes(include=['int64','float64']).columns
cat_col=x.select_dtypes(include='object').columns

preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_col),
        ('dog',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat_col)
    ]
)

pipe=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('model',LogisticRegression(random_state=42,max_iter=1000))
    ]
)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

pipe.fit(x_train,y_train)



jb.dump(pipe,'model/churn_pridiction.pkl')