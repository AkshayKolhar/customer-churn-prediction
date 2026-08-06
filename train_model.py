import pandas as pd 
import matplotlib.pyplot as pt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)

df=pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges']=df['TotalCharges'].replace(' ',np.nan)
df['TotalCharges']=df['TotalCharges'].fillna(0)
df['TotalCharges']=df['TotalCharges'].astype(float)

x=df.drop('Churn',axis=1)
y=df['Churn']
num_col=x.select_dtypes(include=['int64','float64']).columns
cat_col=x.select_dtypes(include=['object','str']).columns

preprocessor=ColumnTransformer(
    transformers=[
        ('num',"passthrough",num_col),
        ('dog',OneHotEncoder(handle_unknown='ignore'),cat_col)
    ]
)
pipe=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('model',LogisticRegression(max_iter=4350))
    ]
)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

pipe.fit(x_train,y_train)
y_pre=pipe.predict(x_test)


print("Accuracy: ",accuracy_score(y_test,y_pre))
print("confusiion Tree :",confusion_matrix(y_test,y_pre))
print(classification_report(y_test,y_pre))


