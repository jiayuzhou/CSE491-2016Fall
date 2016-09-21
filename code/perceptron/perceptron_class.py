# CSE 491 Introduction to Machine Learning
# Python Demo for Perceptron.
#
# By Jiayu Zhou, Sept 20, 2016

from numpy import array, dot
import numpy as np
from random import choice

def train_perceptron(training_data):
    """
    Train a Perceptron model given a set of training data.

    :param training_data: A list of data points, each one is a tuple of (feature vector, label), where labels are +1/-1
    :return: a model vector
    """
    model_size = training_data[0][0].size
    w = np.zeros(model_size)

    ii = 1
    while True:

        # compute results.
        aa = [ np.sign( np.dot(w, xx[0])) * xx[1] for  xx in training_data] 
        # get incorrect predictions.
        indices = [i for i, xx in enumerate(aa) if xx != 1]
        # convergence criteria.
        if not indices:
            break
        # find a wrongly classified sample.
        x_star, y_star = training_data[choice(indices)]
        # update weight via the Perceptron update rule.
        w += y_star * x_star

        ii += 1
    return w


def print_prediction(model, data):
    """
    Print predictions of a model given a dataset.
    :param model: model vector
    :param data: data points.
    :return: nothing.
    """
    for x, _ in data:
        result = dot(x, model)
        print("{}: {} -> {}".format(x[:2], result, np.sign(result)))


if __name__ == '__main__':

    # data sets.
    xor_data = [
        (array([0, 0, 1]), -1),
        (array([0, 1, 1]), 1),
        (array([1, 0, 1]), 1),
        (array([1, 1, 1]), 1)]

    rnd_data = [
        (array([0, 1, 1]), 1),
        (array([0.6, 0.6, 1]), 1),
        (array([1, 0, 1]), 1),
        (array([1, 1, 1]), 1),
        (array([0.3, 0.4, 1]), -1),
        (array([0.2, 0.3, 1]), -1),
        (array([0.1, 0.4, 1]), -1),
        (array([0.5, -0.1, 1]), -1)  # , (array([0.5, 1.0, 1]), -1) # this is an outlier.
    ]

    # set training data.
    train_data = xor_data
    # train_data = rnd_data

    # trained_model = train_perceptron(train_data, use_random=True, iter_eval_func=None)
    trained_model = train_perceptron(train_data)

    # print the model.
    print "Model:", trained_model
    # print the prediction of training data.
    print_prediction(trained_model, train_data)



