import sys
import pandas as pd
import numpy as np


def train(df_training):
    # The method to train the nqaive bayes classifier, by working out the probabilities in the probability table
    numbers_table = np.ones(
        shape=[(df_training.shape[1] - 1) * 2 + 1,
               2])    # set all initial counts to 1 in case a value doesn't appear in the training set
    numbers_table[0][0] += 1 # Initial counts for the actual class counts need to start at 2, as 1 has been added to both the true and false sides
    numbers_table[0][1] += 1

    # This code goes and build the numbers table
    for i in range(df_training.shape[0]):
        if df_training[i][12] == 0:  # if not spam
            numbers_table[0][0] += 1
        elif df_training[i][12] == 1:  # if spam
            numbers_table[0][1] += 1
        for j in range(df_training.shape[1] - 1):  # go from 0 to 11 to exclude class
            if df_training[i][12] == 0:  # if not spam
                if df_training[i][j] == 1:
                    numbers_table[2 * (j + 1) - 1][0] += 1
                elif df_training[i][j] == 0:
                    numbers_table[2 * (j + 1)][0] += 1
            elif df_training[i][12] == 1:  # if spam
                if df_training[i][j] == 1:
                    numbers_table[2 * (j + 1) - 1][1] += 1
                elif df_training[i][j] == 0:
                    numbers_table[2 * (j + 1)][1] += 1

    # This code takes the table of numbers and converts them into percentages of the chance of the thing actually happening
    probability_table = np.ones(
        shape=[(df_training.shape[1] - 1) * 2 + 1, 2])
    for i in range(numbers_table.shape[0]):
        for j in range(numbers_table.shape[1]):
            if i == 0:
                probability_table[i][j] = numbers_table[i][j] / (numbers_table[i][0] + numbers_table[i][1])
            elif i % 2 == 0:
                probability_table[i][j] = numbers_table[i][j] / (numbers_table[i - 1][j] + numbers_table[i][j])
                probability_table[i - 1][j] = numbers_table[i - 1][j] / (numbers_table[i - 1][j] + numbers_table[i][j])
    return probability_table


def test(df_test, probability_table):
    # This code classifies the test instances
    for i in range(df_test.shape[0]):
        prob_not_spam = probability_table[0][0]
        prob_spam = probability_table[0][1]
        for j in range(df_test.shape[1]):
            for k in range(probability_table.shape[1]):
                if df_test[i][j] == 1:
                    if k == 0:
                        prob_not_spam = prob_not_spam * probability_table[2 * (j + 1) - 1][k]
                    elif k == 1:
                        prob_spam = prob_spam * probability_table[2 * (j + 1) - 1][k]
                elif df_test[i][j] == 0:
                    if k == 0:
                        prob_not_spam = prob_not_spam * probability_table[2 * (j + 1)][k]
                    elif k == 1:
                        prob_spam = prob_spam * probability_table[2 * (j + 1)][k]

        # Print out the useful information
        print("Instance " + str(i))
        print("P(C=1, F) = " + str(np.format_float_positional(round(prob_spam, 8))))
        print("P(C=0, F) = " + str(np.format_float_positional(round(prob_not_spam, 8))))
        if prob_spam > prob_not_spam:
            print("The predicted class for this label is Spam (Class 1)")
        else:
            print("The predicted class for this label is NOT Spam (Class 0)")


if __name__ == '__main__':
    arguments = sys.argv[1:]
    # # Load the data
    # df_training = pd.read_csv(arguments[0], sep="     ", header=None, engine="python").to_numpy()
    # df_test = pd.read_csv(arguments[1], sep="     ", header=None, engine="python").to_numpy()
    df_training = pd.read_csv("spamLabelled.dat", sep="     ", header=None, engine="python").to_numpy()
    df_test = pd.read_csv("spamUnlabelled.dat", sep="     ", header=None, engine="python").to_numpy()

    probability_table = train(df_training)

    probability_df = pd.DataFrame(data=probability_table,
                                  index=["P(Class)", "P(Feature 0 = true | Class)", "P(Feature 0 = false | Class)",
                                         "P(Feature 1 = true | Class)", "P(Feature 1 = false | Class)",
                                         "P(Feature 2 = true | Class)", "P(Feature 2 = false | Class)",
                                         "P(Feature 3 = true | Class)", "P(Feature 3 = false | Class)",
                                         "P(Feature 4 = true | Class)", "P(Feature 4 = false | Class)",
                                         "P(Feature 5 = true | Class)", "P(Feature 5 = false | Class)",
                                         "P(Feature 6 = true | Class)", "P(Feature 6 = false | Class)",
                                         "P(Feature 7 = true | Class)", "P(Feature 7 = false | Class)",
                                         "P(Feature 8 = true | Class)", "P(Feature 8 = false | Class)",
                                         "P(Feature 9 = true | Class)", "P(Feature 9 = false | Class)",
                                         "P(Feature 10 = true | Class)", "P(Feature 10 = false | Class)",
                                         "P(Feature 11 = true | Class)", "P(Feature 11 = false | Class)"],
                                  columns=["Not Spam", "Spam"])
    print(probability_df)
    test(df_test, probability_table)
