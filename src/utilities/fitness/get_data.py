import io
import warnings
from os import path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from algorithm.parameters import params
from scipy.io import arff


def impute_missing_values(train_set, test_set):
    """
    This function impute missing values in the provided datasets. For categorical values, it uses the mode.
    For numerical values, it uses the median. In addition, datasets are extended with indicators of missing values,
    since this could be relevant information for the inducer (classifier). And also, it drops features with just one
    value.

    :param train_set: dataset to be used for training. Imputing values are extracted from here
    :param test_set: another dataset where missing values should also be imputed
    :return: copies of train_set and test_set with missing values imputed and columns with missing values indicators.
    """

    def get_impute_value(a_series):
        """
        This function returns the mode or the median to be used to impute missing values. It returns the mode
        in case of categorical features, and the median otherwise

        :param a_series: A pandas Series with the data
        :return: the value to be used to impute missing values
        """
        if a_series.dtype == 'object' or a_series.dtype == 'category' or a_series.dtype == 'bool':
            return a_series.mode()
        else:
            return a_series.median()

    # Obtain the values to be used to impute missing values
    imputing_values = train_set.apply(get_impute_value)
    train_nullvalues = train_set.isnull()

    # Impute the missing values in a copy of the training dataset
    train_set = train_set.copy()
    if type(imputing_values) == pd.DataFrame:
        train_set.fillna(imputing_values.iloc[0, :], inplace=True)
    else:
        train_set.fillna(imputing_values, inplace=True)

    # We also append columns with indicators for missing values. These perhaps were used by classifiers
    labels = list(train_set.columns) + [i + '_missing' for i in train_nullvalues.columns]
    train_set = pd.concat([train_set, train_nullvalues], axis=1)
    train_set.columns = labels

    # Do the same on the test dataset, but with the values selected previously
    if test_set is not None:
        test_nullvalues = test_set.isnull()
        test_set = test_set.copy()
        if type(imputing_values) == pd.DataFrame:
            test_set.fillna(imputing_values.iloc[0, :], inplace=True)
        else:
            test_set.fillna(imputing_values, inplace=True)
        test_set = pd.concat([test_set, test_nullvalues], axis=1, keys=labels)
        test_set.columns = labels

    # Drop columns with just one value
    for col in train_set.columns:
        if len(train_set[col].unique()) == 1:
            train_set.drop(col, inplace=True, axis=1)

            if test_set is not None:
                test_set.drop(col, inplace=True, axis=1)

    return train_set, test_set


# noinspection PyPep8Naming
def get_Xy_train_test_separate(train_filename, test_filename, skip_header=0):
    """
    Read in training and testing data files, and split each into X
    (all columns up to last) and y (last column). The data files should
    contain one row per training example.
    
    :param train_filename: The file name of the training dataset.
    :param test_filename: The file name of the testing dataset.
    :param skip_header: The number of header lines to skip.
    :return: Parsed numpy arrays of training and testing input (x) and
    output (y) data.
    """

    if params['DATASET_DELIMITER']:
        # Dataset delimiter has been explicitly specified.
        delimiter = params['DATASET_DELIMITER']

    else:
        # Try to auto-detect the field separator (i.e. delimiter).
        f = open(train_filename)
        for line in f:
            if line.startswith("#") or len(line) < 2:
                # Skip excessively short lines or commented out lines.
                continue

            else:
                # Set the delimiter.
                if "\t" in line:
                    delimiter = "\t"
                    break
                elif "," in line:
                    delimiter = ","
                    break
                elif ";" in line:
                    delimiter = ";"
                    break
                elif ":" in line:
                    delimiter = ":"
                    break
                else:
                    print(
                        "Warning (in utilities.fitness.get_data.get_Xy_train_test_separate)\n"
                        "Warning: Dataset delimiter not found. "
                        "Defaulting to whitespace delimiter.")
                    delimiter = " "
                    break
        f.close()

    # Read in all training data.
    # train_Xy = np.genfromtxt(train_filename, skip_header=skip_header,
    #                          delimiter=delimiter)
    train_Xy = pd.read_csv(train_filename,
                           delimiter=delimiter)

    try:
        if 'DATASET_TARGET_COLUMN' in params:
            input_indexes = [i != params['DATASET_TARGET_COLUMN'] for i in list(range(train_Xy.shape[1]))]
            train_X = train_Xy.loc[:, input_indexes]
            train_y = train_Xy.loc[:, [not i for i in input_indexes]]
        else:
            # Separate out input (X) and output (y) data.
            train_X = train_Xy.drop(train_Xy.columns[-1], axis='columns')  # all columns but last
            train_y = train_Xy.iloc[:, -1]  # last column

    except IndexError:
        s = "utilities.fitness.get_data.get_Xy_train_test_separate\n" \
            "Error: specified delimiter '%s' incorrectly parses training " \
            "data." % delimiter
        raise Exception(s)

    if test_filename:
        # Read in all testing data.
        test_Xy = np.genfromtxt(test_filename,
                                delimiter=delimiter)

        # Separate out input (X) and output (y) data.
        test_X = test_Xy[:, :-1]  # all columns but last
        test_y = test_Xy[:, -1]  # last column

    else:
        test_X, test_y = None, None

    return train_X, train_y, test_X, test_y


def read_arff(file):
    """
    Read an arff dataset and returns the input variables and output ones

    :param file: Path to the file to be read
    :return: The parsed data contained in the dataset file in a tuple (input, output, metadata)
       . input contains the in features, and output the target, assumed it was the last one.
       . metadata contains information of the features (if categorical, numerical...)
    """

    try:  # arff.arffread fails in case the file has some special characters.
        data, metadata = arff.arffread.loadarff(file)
    except UnicodeEncodeError:  # In such case, we test replacing spanish tildes and loading again
        with open(file, 'r') as f:
            content = ''.join(f.readlines())
            content = content.replace('á', 'a')
            content = content.replace('é', 'e')
            content = content.replace('í', 'i')
            content = content.replace('ó', 'o')
            content = content.replace('ú', 'u')
            content = content.replace('ñ', 'n')
            with io.StringIO(content) as f2:
                data_metadata = arff.loadarff(f2)
                data, metadata = data_metadata

    data = pd.DataFrame(data)
    data = \
        data.apply(lambda x: x.str.decode('utf-8') if x.dtype == 'object' else x)
    data[data == '?'] = np.nan
    num_in_features = data.shape[1] - 1
    input_data = data.iloc[:, :num_in_features]
    output = data.iloc[:, num_in_features]

    return input_data, output, metadata


def read_dataset(filename):
    """
    This file reads a dataset from a file. It decides if it should be read as an ARFF or CSV file.

    :param filename: the name of the file
    :return: a pandas DataFrame with the inputs features of the dataset, a pandas Series with the
        values of the target feature, and a structure with the metadata of the dataset, which
        is only valid for ARFF files. Otherwise, it is None
    """
    metadata = None

    if filename.endswith('.arff'):
        training_in, training_out, metadata = read_arff(filename)
    else:
        # The following instruction is maintained to not to become a change from original PonyGE2,
        # although it is repetitive in case of passing a test dataset
        training_in, training_out, test_in, \
            test_out = get_Xy_train_test_separate(filename, None, skip_header=1)
        training_in = pd.DataFrame(training_in)
        training_in.columns = training_in.columns.astype(str)
        training_out = training_out #.iloc[:,0]

    return training_in, training_out, metadata


def get_data(train, test):
    """
    Return the training and test data for the current experiment. It considers that
    test might be another filename or the index for a cross-validation setting.
    In such case, this fold is removed from the training dataset.
    In addition, it checks the parameter 'IMPUTE_MISSING' to impute missing values
    
    :param train: The desired training dataset filename.
    :param test: The desired testing dataset. If filename, then read it; if int, a fold in previous dataset
    :return: The parsed data contained in the dataset files.
    """

    # 1. Read training dataset
    # Get the path to the training dataset.
    train_set = path.join("..", "datasets", train)
    training_in, training_out, metadata = read_dataset(train_set)

    # 2. Read test dataset, if any
    test_in, test_out = None, None
    if test and isinstance(test, str) and path.isfile(path.join("..", "datasets", test)):
        # Get the path to the testing dataset.
        test_set = path.join("..", "datasets", test)
        test_in, test_out, metadata_test = read_dataset(test_set)
    elif test:
        try:  # Perhaps, test is the string '1', or any other number, so it should be transformed
            # but, perhaps is already an int, and eval(1) fails
            test = eval(test)
        except TypeError:
            pass
        if isinstance(test, int):
            test_in, test_out, training_in, training_out = cross_validation_split(training_in, training_out, test)
        else:
            raise Exception('Test set option unrecognized: ' + test)

    # 3. Impute missing data, in case
    if params.get('IMPUTE_MISSING', False):
        training_in, test_in = impute_missing_values(training_in, test_in)

    return training_in, training_out, test_in, test_out, metadata


def cross_validation_split(inputs, target_values, test: int):
    """
    This function splits the dataset into training and testing according to a
    cross-validation context.

    :param inputs: the input features of the dataset
    :param target_values: the target feature
    :param test: The index of the fold to be used for testing
    :return: a tuple with
        . the testing dataset input features
        . the testing dataset target values
        . the training dataset input features
        . the training dataset target values
    """

    # Check that mandatory parameters are in the parameters file and seed for the random split
    assert 'CROSS_VALIDATION' in params, 'Missing CROSS_VALIDATION parameter (folds),' \
                                         ' mandatory when DATASET_TEST is an integer (' + str(test) + ')'
    if 'CROSS_VALIDATION_SEED' not in params:
        random_state = None
        warnings.warn('Missing CROSS_VALIDATION_SEED parameter in a cross-validation context.\n'
                      'This prevents subsequent executions from using the same cross-validation context, '
                      'because sklearn.StratifiedKFold does not recall how patterns were sorted.')
    else:
        random_state = params['CROSS_VALIDATION_SEED']

    # The real split
    folds_generator = StratifiedKFold(n_splits=params['CROSS_VALIDATION'], shuffle=True, random_state=random_state)
    train_index, test_index = None, None
    for _, (train_index, test_index) in zip(range(test + 1), folds_generator.split(inputs, target_values)):
        pass

    test_in, test_out = inputs.iloc[test_index], target_values[test_index]
    inputs, target_values = inputs.iloc[train_index], target_values[train_index]
    return test_in, test_out, inputs, target_values
