import pandas as pd
import numpy as np
from datasets.Black_box_models import random_forest

from sklearn.model_selection import KFold

#Load data as .csv
df = pd.read_csv("datasets/Diabetes/diabetes.csv")
features = list(df.columns[:-1])
print("Features", features)
label = df.columns[-1]
np_df = df.to_numpy()
patterns = np_df[ : ,:-1]
y = np_df[ : ,-1]
print("Patrones", patterns.shape)
print("Etiquetas", y.shape)

#Do the kfold data partition
kf = KFold(n_splits = 10)

for train_index, test_index in kf.split(patterns):
    print("Train index", type(train_index))
    patterns_train, patterns_test = patterns[train_index], patterns[test_index]
    y_train, y_test = y[train_index], y[test_index]

    predictions = random_forest.rf_train(patterns_train, y_train, features, label)