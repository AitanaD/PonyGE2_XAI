import statistics as st

import numpy as np

from utilities.stats import variables


def write_result_file():
    with open('results_files/' + str(variables.opcion_ejec) + '/' + str(variables.opcion_ejec) + '_' + str(variables.modelo_bb) + '.csv', 'a') as f:

        f.write("Dataset," + "Mean F1 BB train," + "Var F1 BB train," + "Mean F1 BB test," + "Var F1 BB test," + "Mean F1 train," + "Var F1 train," + "Mean F1 test," + "Var F1 test\n")

        mean_f1_bb_train, mean_f1_bb_test, var_f1_bb_train, var_f1_bb_test, mean_f1_pony_train, mean_f1_pony_test, var_f1_pony_train, var_f1_pony_test = compute_values()

        f.write(str(variables.dataset_name[:-4]) + "," + str(mean_f1_bb_train) + "," + str(var_f1_bb_train) + "," + str(mean_f1_bb_test) + "," + str(var_f1_bb_test)
                + "," + str(mean_f1_pony_train) + "," + str(var_f1_pony_train) + "," + str(mean_f1_pony_test) + "," + str(var_f1_pony_test) + '\n')

def write_ejecs_file():
    with open('results_files/' + str(variables.opcion_ejec) + '/' + str(variables.dataset_name[:-4]) + '_' + str(variables.opcion_ejec) + '_' + str(variables.modelo_bb) + '.csv', 'a') as f:

        f.write("Datos," + "Iteracioes\n")
        f.write("F1_bb_train," + str(variables.train_bb_f1) + "\n")
        f.write("F1_bb_test," + str(variables.test_bb_f1) + "\n")
        f.write("F1_ponyGE_train," + str(variables.train_pony_f1) + "\n")
        f.write("F1_ponyGE_test," + str(variables.test_pony_f1) + "\n")

def compute_values():

    mean_f1_bb_train = None
    mean_f1_bb_test = None
    var_f1_bb_train = None
    var_f1_bb_test = None


    if variables.opcion_ejec != "Solo_pony":

        mean_f1_bb_train = round(np.mean(variables.train_bb_f1), 3)
        mean_f1_bb_test = round(np.mean(variables.test_bb_f1), 3)
        var_f1_bb_train = round(np.var(variables.train_bb_f1), 3)
        var_f1_bb_test = round(np.var(variables.test_bb_f1), 3)

    mean_f1_pony_train = round(np.mean(variables.train_pony_f1), 3)
    mean_f1_pony_test = round(np.mean(variables.test_pony_f1), 3)
    var_f1_pony_train = round(np.var(variables.train_pony_f1), 3)
    var_f1_pony_test = round(np.var(variables.test_pony_f1), 3)

    return mean_f1_bb_train, mean_f1_bb_test, var_f1_bb_train, var_f1_bb_test, mean_f1_pony_train, mean_f1_pony_test, var_f1_pony_train, var_f1_pony_test