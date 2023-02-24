import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator
from sklearn.metrics import accuracy_score, f1_score


def rf_train(train_patterns, train_labels, test_labels,  features, label, n_splits, bb_f1:list):
    h2o.init()  # Model object

    # Preprocess train
    """train_path = "../Iris/Train.csv"
    train_df=pd.read_csv(train_path)

    train_df=train_df.drop(columns=['# '])"""

    h2o_train_df = h2o.H2OFrame(train_patterns)
    h2o_train_df = h2o_train_df.cbind(h2o.H2OFrame(train_labels))
    #Add header
    h2o_train_df.col_names = features+[label]

    # Define model
    rf = H2ORandomForestEstimator(ntrees=500)

    # Train model
    rf.train(x=features, y=label, training_frame=h2o_train_df)

    # Model performance (metrics)
    performance = rf.model_performance(test_data=h2o_train_df)
    print(performance)

    # Generate predictions on a validation set:
    pred = rf.predict(test_data=h2o_train_df) >= 0.5

    pred = h2o.H2OFrame(train_patterns).cbind(pred)
    pred.col_names = h2o_train_df.col_names

    # Create csv with predictions --> Transform from h2oframe to csv
    h2o.export_file(pred, "datasets/Black_box_models/prediction/rf_pred"+str(n_splits)+".csv", force=True)


    """Importamos el fichero con las nuevas predicciones
    y medimos el accuracy del modelo"""
    file = "datasets/Black_box_models/prediction/rf_pred"+str(n_splits)+".csv"
    df = pd.read_csv(file)
    array = df.to_numpy()
    bb_f1.append( f1_score(test_labels, array[ : ,-1]))
    print("F1 Score test;", bb_f1[-1])

    # Save the model
    model_path = h2o.save_model(model=rf, path="../../Models", force=True)
    print(model_path)

    return pred
