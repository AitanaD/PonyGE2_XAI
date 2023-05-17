#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params
import sys
from utilities.stats import trackers
from utilities.stats import variables
import builtins


def mane():

    if variables.opcion_ejec == "Solo_pony":
        data_path_train = "train_folds/train"
        data_path_test = "test_folds/test"

    elif variables.opcion_ejec == "PonyOriginal":
        data_path_train = "prediction_train_noprocess/bb_pred"
        data_path_test = "prediction_test_noprocess/bb_pred"

    else:
        data_path_train = "prediction_train/bb_pred"
        data_path_test = "prediction_test/bb_pred"


    """ Run program """
    path = 'parameters/classification.txt'


    parameters = f"""CACHE:                  True
CODON_SIZE:             100000
CROSSOVER:              variable_onepoint
CROSSOVER_PROBABILITY:  0.8
DATASET_TRAIN:          Black_box_models/{data_path_train}{variables.iteracion}.csv
DATASET_TEST:           Black_box_models/{data_path_test}{variables.iteracion}.csv
DEBUG:                  False
ERROR_METRIC:           f1_score
GENERATIONS:            50
MAX_GENOME_LENGTH:      500
GRAMMAR_FILE:           supervised_learning/cec_experiments/tree_complete_grammar_no_arit_expressions.bnf
INITIALISATION:         PI_grow
INVALID_SELECTION:      False
MAX_INIT_TREE_DEPTH:    10
MAX_TREE_DEPTH:         17
MUTATION:               int_flip_per_codon
POPULATION_SIZE:        100
FITNESS_FUNCTION:       supervised_learning.classification
REPLACEMENT:            generational
SELECTION:              tournament
TOURNAMENT_SIZE:        2
VERBOSE: 				False
FILE_PATH:              results/"""
    with open(path, 'w') as f:
        f.write(parameters)

    set_params(['--parameters', 'classification.txt'])  # exclude the ponyge.py arg itself

    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)

    print("#####################################################################################################################")

    trackers.reset_trackers()

    """for variable in dir(builtins):
        if variable not in variables_a_mantener:
            delattr(builtins, variable)"""

if __name__ == "__main__":
    mane()
