import os

import pandas as pd
import numpy as np
import sys

from datasets import blackbox_process
from utils import write_result_file, write_ejecs_file
from sklearn.model_selection import StratifiedKFold
from utilities.stats import variables
from src import  ponyge


base_datasets = ["blood-transfusion.csv", "climate-model.csv", "diabetes.csv", "ilpd.csv", "ozone-level-8hr.csv"]
new_datasets = ["new_ilpd.csv", "new_diabetes.csv", "new_qsar-biodeg.csv", "new_splice.csv", "new_vehicle.csv"]


""" 
    3 opciones de ejecución :
        - Solo_pony : Solo se invoca al algoritmo PonyGE sin las salidas del modelo de caja negra
        - Pre-procesado : Se ejecuta el algoritmo completo pero con los datasets preprocesados
        - Original : Se ejecuta el algoritmo compelto con los datos sin preprocesar
        - PonyOriginal : Los datos ingestados en el algor'itmo PonyGE no estan procesado, sin embargo los del modelo de caja negra si
"""

variables.opcion_ejec = str(sys.argv[1])
print("Opción de ejecución :", variables.opcion_ejec)

""" 
    Se puede indicar un unico modelo de caja negra a la vez.
        - RF : Random Forest
        - GB : Gradient Boosting
        - DL : Deep Neural Network
"""

variables.modelo_bb = str(sys.argv[2])
print("Modelo de caja negra :", variables.modelo_bb)

""" Seleccionar los conjuntos de datos procesados o no, en función
    de la opción elegida"""

if variables.opcion_ejec == "Pre-procesado":
    dir = "datasets/procesados/"

elif variables.opcion_ejec == "PonyOriginal":
    dir = "datasets/procesados/"
    dir_pony = "datasets/originales/"
else:
    dir = "datasets/originales/"


#       """       BUCLE EXTERNO       """

"""os.listdir(dir)"""
for d in new_datasets:

    print(f"*** DATASET {d} ***")
    variables.dataset_name = d

    """Cargar los datos como .csv y asegurar que las etiquetas 
        son 0 ó 1 en todos los datasets"""

    df = pd.read_csv(dir + d)
    unique_values = np.unique(df[df.columns[-1]])
    df[df.columns[-1]] = df[df.columns[-1]].replace({
        unique_values[0]: 0,
        unique_values[1]: 1
    })


    features = list(df.columns[:-1])
    label = df.columns[-1]

    """Convertir el dataframe a np array"""
    np_df = df.to_numpy()
    patterns = np_df[ : ,:-1]
    y = np_df[ : ,-1]

    """ Repetir el proceso para el dataset sin preprocesar si asi se requiere"""
    if variables.opcion_ejec == "PonyOriginal":
        df_pony = pd.read_csv(dir_pony + d[4:])

        """Convertir el dataframe a np array"""
        np_df_pony = df_pony.to_numpy()
        patterns_original = np_df_pony[:, :-1]
        y_original = np_df_pony[:, -1]


    """ Crear las instancias con los K-folds """
    kf = StratifiedKFold(shuffle=True, n_splits=10, random_state=1)


    #       """       BUCLE INTERNO       """

    for train_index, test_index in kf.split(patterns, y):
        print(f"*** FOLD {variables.iteracion + 1} DE {kf.n_splits} ***\n")

        patterns_train, patterns_test = patterns[train_index], patterns[test_index]
        y_train, y_test = y[train_index], y[test_index]

        """ Guardar las particiones de entrenamiento """
        array_data = df.iloc[train_index, :]
        train_data = pd.DataFrame(array_data, columns=df.columns, index=None)
        train_data.to_csv('datasets/Black_box_models/train_folds/train' + str(variables.iteracion) + '.csv')

        """ Guardar las particiones de test """
        array_data = df.iloc[test_index,:]
        test_data = pd.DataFrame(array_data, columns=df.columns, index=None)
        test_data.to_csv('datasets/Black_box_models/test_folds/test'+str(variables.iteracion)+'.csv')

        """ Entrenar el modelo de caja negra indicado"""

        if variables.opcion_ejec != "Solo_pony":
            predictions = blackbox_process.blackbox_process(patterns_train, y_train, patterns_test, y_test, features, label)

        """ Sustituir los patrones preprocesados por los originales, si se requiere"""
        if variables.opcion_ejec == "PonyOriginal":
            patterns_train_org, patterns_test_org = patterns_original[train_index], patterns_original[test_index]

            """ Guardar las particiones de entrenamiento """
            X_train_data = pd.DataFrame(patterns_train_org, columns=df_pony.columns[:-1], index=None)
            process_train_data = pd.read_csv('datasets/Black_box_models/prediction_train/bb_pred' + str(variables.iteracion) + '.csv')
            new_train_data = pd.concat([X_train_data, process_train_data[process_train_data.columns[-1]]], axis=1)
            new_train_data.to_csv('datasets/Black_box_models/prediction_train_noprocess/bb_pred' + str(variables.iteracion) + '.csv')

            """ Guardar las particiones de test """
            X_test_data = pd.DataFrame(patterns_test_org, columns=df_pony.columns[:-1], index=None)
            process_test_data = pd.read_csv('datasets/Black_box_models/prediction_test/bb_pred' + str(variables.iteracion) + '.csv')
            new_test_data = pd.concat([X_test_data, process_test_data[process_test_data.columns[-1]]], axis=1)
            new_test_data.to_csv('datasets/Black_box_models/prediction_test_noprocess/bb_pred' + str(variables.iteracion) + '.csv')

        """ Invocar al algoritmo de ponyGE"""
        ponyge.mane()

        variables.iteracion = variables.iteracion + 1

    """ Almacenar la media y desviación tipica de los resultados
        en un .csv"""
    write_result_file()
    write_ejecs_file()

    """ Resetear las listas de resultados"""
    variables.test_bb_f1 = []
    variables.train_bb_f1 = []
    variables.test_pony_f1 = []
    variables.train_pony_f1 = []
    variables.iteracion = 0
