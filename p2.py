# %%
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_squared_error, jaccard_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %%


def p_i():
    categoryColumns = ["Sex", "ChestPainType", "RestingECG",
                       "ExerciseAngina", "ST_Slope"]
    categoryColumnsDict = {i: "category" for i in categoryColumns}
    # (i)
    df = pd.read_csv('./data/heart.csv', dtype=categoryColumnsDict).dropna()
    df[categoryColumns] = df[categoryColumns].apply(
        lambda col: pd.Categorical(col).codes)
    y_all = df['HeartDisease']
    X_all = df.loc[:, df.columns != 'HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=1)
    # %%
    # Note on accuracy
    # https://developers.google.com/machine-learning/crash-course/classification/accuracy

    # Jaracard Score
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score
    min_samples = list(range(1, 26))
    accuracy_array = []
    for i in min_samples:
        clf = tree.DecisionTreeClassifier(
            min_samples_leaf=i).fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        accuracy_array.append(jaccard_score(y_val, y_val_pred))

    fig, ax = plt.subplots()
    ax.plot(min_samples, accuracy_array)
    ax.grid(True)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Min Samples per Leaf")
    ax.set_title('Validation Accuracy vs. Min Samples per Leaf')
    plt.savefig('./images/P2_ii_AccuracyvsSamples')

    # %%

    # (iii)
    bestMinNumberSamples = min_samples[accuracy_array.index(
        max(accuracy_array))]
    print(bestMinNumberSamples)
    # ~ 9 is the best miniumum number of leaf node observations

    clf = tree.DecisionTreeClassifier(
        min_samples_leaf=bestMinNumberSamples).fit(X_test, y_test)
    y_test_pred = clf.predict(X_test)
    test_accuracy = jaccard_score(y_test, y_test_pred)
    print(test_accuracy)
    # Best accuracy 83%

    test_confusion_matrix = confusion_matrix(
        y_test, y_test_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=test_confusion_matrix, display_labels=clf.classes_)
    disp.plot()

# p_i()


def p_iv():
    categoryColumns = ["Sex", "ChestPainType", "RestingECG",
                       "ExerciseAngina", "ST_Slope"]
    categoryColumnsDict = {i: "category" for i in categoryColumns}
    # (i)
    df = pd.read_csv('./data/heart.csv', dtype=categoryColumnsDict).dropna()
    df[categoryColumns] = df[categoryColumns].apply(
        lambda col: pd.Categorical(col).codes)
    y_all = df['RestingECG']
    X_all = df.loc[:, df.columns != 'RestingECG']

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=1)

    min_samples = list(range(1, 26))
    accuracy_array = []
    for i in min_samples:
        clf = tree.DecisionTreeClassifier(
            min_samples_leaf=i).fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        accuracy_array.append(accuracy_score(y_val, y_val_pred))

    fig, ax = plt.subplots()
    ax.plot(min_samples, accuracy_array)
    ax.grid(True)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Min Samples per Leaf")
    ax.set_title('Validation Accuracy vs. Min Samples per Leaf')
    plt.savefig('./images/P2_iv_AccuracyvsSamples')

    bestMinNumberSamples = min_samples[accuracy_array.index(
        max(accuracy_array))]
    print(bestMinNumberSamples)

    clf = tree.DecisionTreeClassifier(
        min_samples_leaf=bestMinNumberSamples).fit(X_test, y_test)
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(test_accuracy)

    test_confusion_matrix = confusion_matrix(
        y_test, y_test_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=test_confusion_matrix, display_labels=clf.classes_)
    disp.plot()


p_iv()
