# CSE 491 Introduction to Machine Learning
# Python Demo for Linear Regression.
#
# By Jiayu Zhou, Oct 11, 2016

import time
import numpy as np
import matplotlib.pyplot as plt

def rand_split_train_test(data, label, train_perc):
    """
    Randomly split training and testing data by specify a percentage of training.
    :param data: X
    :param label: y
    :param train_perc: training percentage
    :return: training X, testing X, training y, testing y.
    """
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    idx = np.random.permutation(data.shape[0])
    idx_tr = idx[0: train_sample]
    idx_te = idx[train_sample:]

    data_tr = data[idx_tr, :]
    data_te = data[idx_te, :]
    label_tr = label[idx_tr, :]
    label_te = label[idx_te, :]

    return data_tr, data_te, label_tr, label_te


def subsample_data(data, label, subsample_size):
    """
    Subsample a portion of data
    :param data: X
    :param label: y
    :param subsample_size: size of the subsample.
    :return: sampled X, sampled y
    """
    # protected sample size
    subsample_size = np.max([1, np.min([data.shape[0], subsample_size])])

    idx = np.random.permutation(data.shape[0])
    idx = idx[0: subsample_size]
    data = data[idx, :]
    label = label[idx, :]
    return data, label


def generate_rnd_data(feature_size, sample_size, bias=False):
    """
    Generate random data
    :param feature_size: number of features
    :param sample_size:  number of sample size
    :param bias:  do we include an extra bias term and .
    :return: data (sample_size X feature_size), label (sample_size X 1), truth_model (feature_size X 1)
    """
    # Generate X matrix.
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) \
        if bias else np.random.randn(sample_size, feature_size)  # the first dimension is sample_size (n X d)

    # Generate ground truth (oracle) model.
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10

    # Generate label.
    label = np.dot(data, truth_model)

    # add element-wise gaussian noise to each label.
    label += np.random.randn(sample_size, 1)
    return data, label, truth_model


def mean_squared_error(true_label, predicted_label):
    """
    Compute the mean squared error given a set of predictive values and their ground truth.
    :param true_label: true target
    :param predicted_label: predicted target
    :return: mean squared error.
    """
    return np.sqrt(np.sum((true_label - predicted_label)**2)/true_label.size)


def least_squares(feature, target):
    """
    Compute least squares using closed form
    :param feature: X
    :param target: y
    :return: computed weight vector
    """
    return np.dot(np.linalg.inv(np.dot(feature.T, feature)), np.dot(feature.T, target))



def least_squares_gd(feature, target, max_iter=1000, step_size=0.00001, tol=1e-3):
    feature_dim = feature.shape[1]
    weight = np.zeros((feature_dim, 1))
    obj_val = []
    for i in range(max_iter):
        weight_old = weight.copy()

        err = np.dot(feature, weight) - target
        obj_val += [np.linalg.norm(err, 'fro')**2/2]
        grad = np.dot(feature.T, err)
        weight -= step_size * grad

        # stop criteria.
        if np.linalg.norm(weight - weight_old, 'fro') < tol:
            break
    return weight, obj_val


def ridge_regression(feature, target, lam=1e-17):
    """
    Compute ridge regression using closed form
    :param feature: X
    :param target: y
    :param lam: lambda
    :return:
    """
    feature_dim = feature.shape[1]
    return np.dot(np.linalg.inv(np.dot(feature.T, feature) + np.eye(feature_dim)*lam), np.dot(feature.T, target))




def exp1():
    """
    Simple training and testing to make least squares and ridge regression work.
    :return: nothing
    """
    # EXP1: training testing.
    # generate a data set.
    (feature_all, target_all, model) = generate_rnd_data(feature_size=3, sample_size=20, bias=True)
    #print np.concatenate((feature_all, target_all), axis=1)
    # split training/testing
    feature_train, feature_test, target_train, target_test = rand_split_train_test(feature_all, target_all, train_perc=0.8)
    #print np.concatenate((feature_train, target_train), axis=1)
    #print np.concatenate((feature_test, target_test), axis=1)
    # compute model
    reg_model_lsqr = least_squares(feature_train, target_train)
    reg_model_ridge = ridge_regression(feature_train, target_train, lam=1)

    # evaluate performance
    print 'Training MSE(lsqr):', mean_squared_error(target_train, np.dot(feature_train, reg_model_lsqr))
    print 'Testing MSE(lsqr):', mean_squared_error(target_test, np.dot(feature_test, reg_model_lsqr))
    print 'Training MSE(ridge):', mean_squared_error(target_train, np.dot(feature_train, reg_model_ridge))
    print 'Testing MSE(ridge):', mean_squared_error(target_test, np.dot(feature_test, reg_model_ridge))



def exp2():
    """
    generalization performance: increase sample size.
    :return: nothing
    """
    sample_size_arr = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    (feature_all, target_all, model) = generate_rnd_data(feature_size=100, sample_size=1000, bias=True)
    feature_hold, feature_test, target_hold, target_test = \
        rand_split_train_test(feature_all, target_all, train_perc=0.9)

    perf_train = []
    perf_test = []
    for train_sample_size in sample_size_arr:
        feature_train, target_train = subsample_data(feature_hold, target_hold, train_sample_size)
        reg_model = ridge_regression(feature_train, target_train, lam=1e-5)
        perf_train += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        perf_test += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]

    plt.figure()
    train_plot, = plt.plot(sample_size_arr, np.log10(perf_train), linestyle='-', color='b', label='Training Error')
    test_plot, = plt.plot(sample_size_arr, np.log10(perf_test), linestyle='-', color='r', label='Testing Error')
    plt.xlabel("Sample Size")
    plt.ylabel("Error (log)")
    plt.title("Generalization performance: increase sample size fix dimensionality")
    plt.legend(handles=[train_plot, test_plot])
    plt.show()


def exp3():
    """
    Generalization performance: increase dimensionality.
    :return: nothing
    """
    dimensionality_arr = [100, 150, 200, 250, 300, 350, 400, 450]

    perf_train = []
    perf_test = []
    for dimension in dimensionality_arr:
        (feature_all, target_all, model) = generate_rnd_data(feature_size=dimension, sample_size=1000, bias=True)
        feature_train, feature_test, target_train, target_test = \
            rand_split_train_test(feature_all, target_all, train_perc=0.9)
        reg_model = ridge_regression(feature_train, target_train, lam=1e-5)
        perf_train += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        perf_test += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]

    plt.figure()
    train_plot, = plt.plot(dimensionality_arr, np.log10(perf_train), linestyle='-', color='b', label='Training Error')
    test_plot, = plt.plot(dimensionality_arr, np.log10(perf_test), linestyle='-', color='r', label='Testing Error')
    plt.xlabel("Dimensionality")
    plt.ylabel("Error (log)")
    plt.title("Generalization performance: increase dimensionality fix sample size")
    plt.legend(handles=[train_plot, test_plot])
    plt.show()

def exp4():
    """
    Computational time: increase dimensionality.
    :return:
    """
    dimensionality_arr = range(100, 2000, 100)

    perf_train = []
    perf_test = []
    time_elapse = []
    for dimension in dimensionality_arr:
        (feature_all, target_all, model) = generate_rnd_data(feature_size=dimension, sample_size=1000, bias=True)
        feature_train, feature_test, target_train, target_test = \
            rand_split_train_test(feature_all, target_all, train_perc=0.9)
        t = time.time()
        reg_model = ridge_regression(feature_train, target_train, lam=1e-5)
        time_elapse += [time.time() - t]
        print 'Finished model of dimension {}'.format(dimension)

        perf_train += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        perf_test += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]

    plt.figure()
    time_plot, = plt.plot(dimensionality_arr, time_elapse, linestyle='-', color='r', label='Time cost')
    plt.xlabel("Dimensionality")
    plt.ylabel("Time (ms)")
    plt.title("Computational efficiency.")
    plt.legend(handles=[time_plot])
    plt.show()


def exp5():
    # gradient descent.
    (feature_all, target_all, model) = generate_rnd_data(feature_size=300, sample_size=10000, bias=True)
    feature_train, feature_test, target_train, target_test = \
        rand_split_train_test(feature_all, target_all, train_perc=0.9)

    reg_model = least_squares(feature_train, target_train)
    reg_model_gd, obj_val = least_squares_gd(feature_train, target_train, max_iter=300, step_size=0.00001)

    print 'Model difference {}'.format(np.linalg.norm(reg_model - reg_model_gd, 'fro')/reg_model.size)
    plt.figure()
    plt.plot(range(len(obj_val)), obj_val, linestyle='-', color='r', label='Objective Value')
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Convergence.")
    plt.show()

    lsqr_obj = lambda model: np.linalg.norm(np.dot(feature_train, model) - target_train, 'fro') ** 2 / 2
    print 'Closed Form Objective: ', lsqr_obj(reg_model)
    print 'Gradient Descent Objective: ', lsqr_obj(reg_model)

if __name__ == '__main__':
    plt.interactive(False)

    # set seeds to get repeatable results.
    np.random.seed(491)

    # # EXP1: training testing.
    #exp1()
    #
    # # EXP2: generalization performance: increase sample size.
    # exp2()
    #
    # # EXP3: generalization performance: increase dimensionality.
    # exp3()
    #
    # # EXP4: computational complexity by varing dimensions.
    # exp4()

    # EXP5: gradient descent and convergence.
    exp5()

