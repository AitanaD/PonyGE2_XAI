import statistics as st


def write_result_file(name : str, bb_f1_test: list, f1_test:list, f1_train:list):
    with open('results_files/scores.txt', 'w') as f:
        f.write("Dataset" + "Mean F1 BB test" + "Var F1 BB test" + "Mean F1 test" + "Var F1 test" + "Mean F1 train" + "Var F1 train")

        mean_acc_test, mean_f1_test, mean_f1_train, var_acc_test, var_f1_test = compute_values(bb_f1_test, f1_test, f1_train)

        f.write(str(name) + str(mean_acc_test) + str(var_acc_test) + str(mean_f1_test) + str(var_f1_test) + str(mean_f1_train) + '\n')

    f.close()

def compute_values(bb_f1_test:list, f1_test:list, f1_train:list):

    mean_acc_test  = sum(bb_f1_test) / len(bb_f1_test)
    mean_f1_test = sum(f1_test) / len(f1_test)
    mean_f1_train = sum(f1_train) / len(f1_train)

    var_acc_test = st.variance(bb_f1_test)
    var_f1_test = st.variance(f1_test)

    return mean_acc_test, mean_f1_test, mean_f1_train, var_acc_test, var_f1_test