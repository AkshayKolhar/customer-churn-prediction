from sklearn.ensemble import RandomForestClassifier

from train_model import train_and_pre
import sklearn
print("scikit versiion: ",sklearn.__version__)
rf=RandomForestClassifier(random_state=42)
print(rf)
train_and_pre(RandomForestClassifier(random_state=42))