import h2o
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator

h2o.init()  #Model object

#Preprocess train
train_path = "../datasets/Iris/Train.csv"
train_df=pd.read_csv(train_path)
train_df=train_df.drop(columns=['# '])
h2o_train_df=h2o.H2OFrame(train_df)
features = list(train_df.columns)
label = 'y'

#Preprocess test
test_path = "../datasets/Iris/Test.csv"
test_df=pd.read_csv(test_path)
test_df=test_df.drop(columns=['# '])
h2o_test_df=h2o.H2OFrame(test_df)
features = list(test_df.columns)
label = 'y'

# Define model
model = H2ORandomForestEstimator(ntrees=500)

# Train model
model.train(x=features, y=label, training_frame=h2o_train_df)

# Model performance
performance = model.model_performance(test_data=h2o_test_df)
print(performance)

# Save the model
model_path = h2o.save_model(model=model, path="../Models", force=True)
print(model_path)