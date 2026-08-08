from sklearn.tree import DecisionTreeClassifier
from train_model import train_and_pre

train_and_pre(DecisionTreeClassifier(max_depth=1,random_state=42))