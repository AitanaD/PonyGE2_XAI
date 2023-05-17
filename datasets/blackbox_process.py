import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2ODeepLearningEstimator
from sklearn.metrics import accuracy_score, f1_score
from utilities.stats import variables


def blackbox_process(train_patterns, train_labels, test_patterns,  test_labels,  features, label):

    """ Inicializar el objeto H2O """
    h2o.init()

    """ Definir los modelos de caja negra posibles"""
    bb_option = {
        "RF": H2ORandomForestEstimator(ntrees=100),
        "GB": H2OGradientBoostingEstimator(ntrees=100),
        "DL": H2ODeepLearningEstimator(epochs=1000),
    }

    h2o_train_df = h2o.H2OFrame(train_patterns)
    h2o_train_df = h2o_train_df.cbind(h2o.H2OFrame(train_labels))

    """ Añadir las cabeceras al dataframe """
    h2o_train_df.col_names = features+[label]

    """ Repetir el proceso con los datos de test"""
    h2o_test_df = h2o.H2OFrame(test_patterns)
    h2o_test_df = h2o_test_df.cbind(h2o.H2OFrame(test_labels))
    h2o_test_df.col_names = features + [label]

    """ Inicializar el o los modelos de caja negra """
    bb_model = bb_option[variables.modelo_bb]

    """ Entrenamiento del modelo """
    bb_model.train(x=features, y=label, training_frame=h2o_train_df)

    bb_model.model_performance(test_data=h2o_train_df)

    """ Realizar predicciones sobre los datos de entrenamiento.
        Esto es así para después poder alimentar a PonyGE con
        las predicciones aquí dadas"""

    pred = bb_model.predict(test_data=h2o_train_df) >= 0.5
    pred_df = pred.as_data_frame().to_numpy()

    """ Almacenar los resultados de la f1 en entrenamiento """
    variables.train_bb_f1.append(f1_score(train_labels, pred_df[:, -1], average="weighted"))
    print(f"Resultado en train: {variables.train_bb_f1}")

    pred = h2o.H2OFrame(train_patterns).cbind(pred)
    pred.col_names = h2o_train_df.col_names

    """ Transformar las predicciones de entrenamiento de H2O-frame a .csv"""
    h2o.export_file(pred, "datasets/Black_box_models/prediction_train/bb_pred"+str(variables.iteracion)+".csv", force=True)


    """ Almacenar los resultados de la f1 en test """
    pred_test = bb_model.predict(test_data=h2o_test_df) >= 0.5
    pred_test = pred_test.as_data_frame().to_numpy()

    variables.test_bb_f1.append( f1_score(test_labels, pred_test[:, -1], average="weighted"))
    print(f"Resultado en test: {variables.test_bb_f1}")

    pred = h2o.H2OFrame(test_patterns).cbind(h2o.H2OFrame(pred_test))
    pred.col_names = h2o_test_df.col_names

    """ Transformar las predicciones de entrenamiento de H2O-frame a .csv"""
    h2o.export_file(pred, "datasets/Black_box_models/prediction_test/bb_pred" + str(variables.iteracion) + ".csv",
                    force=True)

    """model_path = h2o.save_model(model=rf, path="../../Models", force=True)
    print(model_path)"""

    print(f"\n*** TERMINADO EL PROCESDO DE CAJA NEGRA ***\n")

    return pred
