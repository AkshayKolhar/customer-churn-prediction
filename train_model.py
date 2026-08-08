import pandas as pd 
import matplotlib.pyplot as pt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder,StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score)

def train_and_pre(model):
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
            ('model',model)
        ]
    )
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    pipe.fit(x_train,y_train)

    print(pipe.classes_)
    y_prob=pipe.predict_proba(x_test)[:,1]

    thresh=[0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
    for t in thresh:
        y_pre=np.where(y_prob>=t,'Yes','No')

        recall=recall_score(y_test,y_pre, pos_label='Yes')
        pre=precision_score(y_test,y_pre, pos_label='Yes')
        f1=f1_score(y_test,y_pre,pos_label='Yes')

        print(f"threshold : {t:.2f} | recall : {recall:.2f} | precison: {pre:.2f} |f1_score: {f1:.2f}")

    Y_tra_pr=pipe.predict(x_train)


    print("Accuracy: ",accuracy_score(y_test,y_pre))
    print("Accuracy on traing : ",accuracy_score(y_train,Y_tra_pr))
    print("confusiion Tree :",confusion_matrix(y_test,y_pre))
    print(classification_report(y_test,y_pre))


