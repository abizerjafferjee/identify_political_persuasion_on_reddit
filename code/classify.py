from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
import argparse
import sys
import os
import numpy as np
import csv

OUTPUT_1 = "3.1.csv"
OUTPUT_2 = "3.2.csv"
OUTPUT_3 = "3.3.csv"
OUTPUT_4 = "3.4.csv"

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    sum_true_true = 0
    for i in range(len(C)):
        sum_true_true += C[i][i]

    sum_all = 0
    for i in range(len(C)):
        for j in range(len(C[i])):
            sum_all += C[i][j]

    accuracy = sum_true_true/sum_all
    return accuracy

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    for i in range(len(C)):
        truly_k = C[i][i]
        all_k = 0
        for j in range(len(C[i])):
            all_k += C[i][j]
        recall = truly_k/all_k
        result.append(recall)
    return result

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    result = []
    for i in range(len(C)):
        truly_k = C[i][i]
        all_k = 0
        for j in range(len(C)):
            all_k += C[j][i]
        precision = truly_k/all_k
        result.append(precision)
    return result

def eval_helper(classifier, matrix):
    """
    calcular accuracy, recall and precision for classifier and return results
    """
    result = []

    if classifier == "svcl":
        result.append(1)
    elif classifier == "svcr":
        result.append(2)
    elif classifier == "rfc":
        result.append(3)
    elif classifier == "mlp":
        result.append(4)
    elif classifier == "ada":
        result.append(5)

    acc = accuracy(matrix)
    result.append(acc)

    recalls = recall(matrix)
    for r in recalls:
        result.append(r)

    precisions = precision(matrix)
    for p in precisions:
        result.append(p)

    for i in matrix:
        for j in i:
            result.append(j)

    return result

def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    csv_file = open(OUTPUT_1, "w")
    csv_writer = csv.writer(csv_file)

    best_acc = []

    features = np.load(filename)
    data = features["arr_0"]

    X = np.array([row[:173] for row in data])
    y = np.array([row[173] for row in data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    # svc linear
    print("scv linear")
    svc_linear = SVC(kernel = "linear")
    svc_linear.fit(X_train, y_train)

    y_pred = svc_linear.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    result = eval_helper("svcl", matrix)
    best_acc.append(result[1])
    csv_writer.writerow(result)

    print("scv rbf")
    # svc radial basis function
    svc_rbf = SVC(kernel = "rbf", gamma = 2)
    svc_rbf.fit(X_train, y_train)

    y_pred = svc_rbf.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    result = eval_helper("svcr", matrix)
    best_acc.append(result[1])
    csv_writer.writerow(result)

    print("random forest")
    # randomforestclassifier
    rfc = RandomForestClassifier(n_estimators = 10, max_depth = 5)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    result = eval_helper("rfc", matrix)
    best_acc.append(result[1])
    csv_writer.writerow(result)

    print("mlp classifier")
    # MLPClassifier
    mlp = MLPClassifier(alpha = 0.05)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    result = eval_helper("mlp", matrix)
    best_acc.append(result[1])
    csv_writer.writerow(result)

    print("ada boost")
    # AdaBoost
    adaboost = AdaBoostClassifier()
    adaboost.fit(X_train, y_train)

    y_pred = adaboost.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)

    result = eval_helper("ada", matrix)
    best_acc.append(result[1])
    csv_writer.writerow(result)

    csv_file.close()

    iBest = 0
    max_acc = 0
    for bi in range(len(best_acc)):
        if best_acc[bi] >= max_acc:
            max_acc = best_acc[bi]
            iBest = bi+1

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test,iBest):

    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    classifiers = {1:SVC(kernel = "linear"), 2:SVC(kernel = "rbf", gamma = 2),
    3:RandomForestClassifier(n_estimators = 10, max_depth = 5),
    4:MLPClassifier(alpha = 0.05), 5:AdaBoostClassifier()}

    clf = classifiers[iBest]

    csv_file = open(OUTPUT_2, "w")
    csv_writer = csv.writer(csv_file)

    # create data sets for the different ranges
    ranges = [1000, 5000, 10000, 15000, 20000]
    batches = []
    for x in ranges:
        batches.append((X_train[:x], y_train[:x]))

    accs = []
    # train different size data sets on best classifier
    print("training different sized batches")
    for batch in batches:
        X_t = batch[0]
        y_t = batch[1]

        clf.fit(X_t, y_t)
        y_pred = clf.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
        accurac = accuracy(matrix)
        accs.append(accurac)

    csv_writer.writerow(accs)
    csv_writer.writerow(["I would expect the accuracy to increase as" +
    "the training set increases in size. A smaller training set" +
    "may overfit the model on the training set causing it to perform" +
    "poorly on test set. The actually trend is as expected."])
    csv_file.close()

    X_1k = batches[0][0]
    y_1k = batches[0][1]

    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    classifiers = {1:SVC(kernel = "linear"), 2:SVC(kernel = "rbf", gamma = 2),
    3:RandomForestClassifier(n_estimators = 10, max_depth = 5),
    4:MLPClassifier(alpha = 0.05), 5:AdaBoostClassifier()}

    csv_file = open(OUTPUT_3, "w")
    csv_writer = csv.writer(csv_file)

    k_val = [5, 10, 20, 30, 40, 50]
    accuracy_1k32k = []
    # get pvals for all k_vals and get accuracy at k=5 for 32k and 1k
    print("extracting best features and evaluating on them")
    for j in k_val:
        selector = SelectKBest(f_classif, k = j)
        X32k_new = selector.fit_transform(X_train, y_train)
        idx_32k = selector.get_support(indices=True)

        pp = selector.pvalues_
        p_vals = [j]
        for id in idx_32k:
            p_vals.append(pp[id])
        csv_writer.writerow(p_vals)

        X1k_new = selector.fit_transform(X_1k, y_1k)
        idx_1k = selector.get_support(indices=True)

        pp = selector.pvalues_
        p_vals = [j]
        for id in idx_1k:
            p_vals.append(pp[id])

        if j == 5:

            Xtest_new = selector.fit_transform(X_test, y_test)

            clf = classifiers[i]
            clf.fit(X1k_new, y_1k)
            y_pred = clf.predict(Xtest_new)
            matrix = confusion_matrix(y_test, y_pred)
            accuracy_1k32k.append(accuracy(matrix))

            clf = classifiers[i]
            clf.fit(X32k_new, y_train)
            y_pred = clf.predict(Xtest_new)
            matrix = confusion_matrix(y_test, y_pred)
            accuracy_1k32k.append(accuracy(matrix))

    csv_writer.writerow(accuracy_1k32k)
    csv_writer.writerow(["Features 149 and 163 are common for both 1k and 32k data" +
    "sizes from k=5 onwards."])
    csv_writer.writerow(["P-values are generally smaller given more data." +
    "More data creates a better distribution to calculate goodness of it."])
    csv_writer.writerow(["The top 5 features for 32k set were 6, 110 (theyliwc)," +
    "143 (geniunereceptivity), 149 (impulsive receptivity), 163 (type_areceptivity)." +
    "110 identifies a specific group (left, right ...), 143 identifies genuine speak" +
    "which could be the specific nature of people classified differently," +
     "149 identifies impulsive speak which again could identify the nature" +
     "of commenters in specific group e.g. right, 163 identifies type which" +
     "could be associated with type of group."])
    csv_file.close()

def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
    '''
    csv_file = open(OUTPUT_4, "w")
    csv_writer = csv.writer(csv_file)

    kf = KFold(n_splits=5, shuffle = True)

    features = np.load(filename)
    data = features["arr_0"]

    X = np.array([row[:173] for row in data])
    y = np.array([row[173] for row in data])

    classifiers = {1:SVC(kernel = "linear"), 2:SVC(kernel = "rbf", gamma = 2),
    3:RandomForestClassifier(n_estimators = 10, max_depth = 5),
    4:MLPClassifier(alpha = 0.05), 5:AdaBoostClassifier()}

    main_results = []

    # run all classifiers for each train test combination
    print("evaluating the folds")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results = []
        for key in classifiers.keys():
            clf = classifiers[key]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            matrix = confusion_matrix(y_test, y_pred)
            accs = accuracy(matrix)
            results.append(accs)

        main_results.append(results)
        csv_writer.writerow(results)

    # create appropriate vector from accuracy results for ttest calculation
    vectors = np.zeros((5, 5))
    for clf in range(5):
        for result in range(5):
            vectors[clf][result] = main_results[result][clf]

    # calculate the ttests
    stat = []
    for j in range(5):
        if j != i-1:
            a= vectors[i-1]
            b = vectors[j]
            S = stats.ttest_rel(a,b)
            stat.append(S)

    csv_writer.writerow(stat)
    csv_writer.writerow(["The best classifier is significantly better than all" +
    "the other classifiers at the 5% level. This validates the finding from 3.1." +
    "This may be because we used k = 5 so the training set size in each" +
    "iteration is 32k which is the same as the training set size in 3.1."])
    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    print("running 3.1")
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    print("running 3.2")
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    print("running 3.3")
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    print("running 3.4")
    class34(args.input, iBest)
