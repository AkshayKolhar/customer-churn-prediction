from sklearn.neighbors import KNeighborsClassifier

from train_model import train_and_pre

train_and_pre(KNeighborsClassifier(n_neighbors=5))