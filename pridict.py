import joblib as jb
import numpy as np

pipe=jb.load("model/churn_pridiction.pkl")
def prediction_churn(details):
    y_prob=pipe.predict_proba(details)[0,1]
    y_pre="Yes " if y_prob>=0.30 else "No"

    return y_prob,y_pre