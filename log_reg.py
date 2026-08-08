from sklearn.linear_model import LogisticRegression
from train_model import (train_and_pre)

train_and_pre(LogisticRegression(max_iter=1000,random_state=42))
