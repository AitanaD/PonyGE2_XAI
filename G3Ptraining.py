import pandas as pd
from sklearn import preprocessing

from datasets.Black_box_models import random_forest
from utils import write_result_file
from sklearn.model_selection import KFold

#TODO: Repeat process on all datasets

bb_f1 = list()
index = 0

#Load data as .csv
df = pd.read_csv("datasets/new_diabetes.csv")
features = list(df.columns[:-1])
print("Features", features)
label = df.columns[-1]

#Convert to np array
np_df = df.to_numpy()
patterns = np_df[ : ,:-1]
y = np_df[ : ,-1]
print("Patrones", patterns.shape)
print("Etiquetas", y.shape)

#Normalize patterns
norm_patterns = preprocessing.normalize(patterns)
print("Normalizados", norm_patterns)

#Do the kfold data partition
kf = KFold(n_splits = 10)

for train_index, test_index in kf.split(patterns):
    print("Train index", type(train_index))
    print("Iteraciones:", kf.split(patterns))
    patterns_train, patterns_test = patterns[train_index], patterns[test_index]
    y_train, y_test = y[train_index], y[test_index]


    #Save train partition
    array_data = df.iloc[test_index,:]
    test_data = pd.DataFrame(array_data, columns=df.columns, index=None)
    test_data.to_csv('datasets/Black_box_models/test_folds/test'+str(index)+'.csv')

    predictions = random_forest.rf_train(patterns_train, y_train, y_test, features, label, index, bb_f1)
    index = index + 1

    #TODO: Implement call to PonyGE and save results

    #TODO: Save scores in a file
    write_result_file(name, bb_f1, pg2_f1_test, pg2_f1_train)