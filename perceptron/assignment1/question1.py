import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_dir = "./Datasets-Question1"
tr_data_str = "./Datasets-Question1/dataset{}/Train{}.csv"
ts_data_str = "./Datasets-Question1/dataset{}/Test{}.csv"
output_dir = "./output/question1"
num_dataset = len(next(os.walk(data_dir))[1])
features = ["x", "y"]
color = ["red", "blue"]
marker = [".", "+"]
max_itr = 100000

os.makedirs(output_dir, exist_ok=True)


def get_y(x, w):
    return -1 * (w[0] + w[1]*x) / w[2]


for i in range(1, num_dataset+1):
    tr_data_file = tr_data_str.format(i, i)
    ts_data_file = ts_data_str.format(i, i)

    print("========= D A T A S E T {} =========".format(i))
    print("Training data file: {}".format(tr_data_file))
    print("Test data file: {}".format(ts_data_file))

    # Reading the data from the file
    tr_df = pd.read_csv(tr_data_file, header=None)
    ts_df = pd.read_csv(ts_data_file, header=None)

    # Converting into numpy array
    tr_labels = tr_df[2].add(0.1).astype(np.int).to_numpy()
    tr_data = tr_df[[0, 1]].to_numpy()

    ts_labels = ts_df[2].add(0.1).astype(np.int).to_numpy()
    ts_data = ts_df[[0, 1]].to_numpy()

    # Augmenting the data and mapping labels {0, 1} to {-1, +1}
    X = np.hstack((np.ones(tr_data.shape[0]).reshape(-1, 1), tr_data))
    Y = tr_labels * 2 - 1

    X_test = np.hstack((np.ones(ts_data.shape[0]).reshape(-1, 1), ts_data))
    Y_test = ts_labels * 2 - 1

    # Weights initialization W = [w0, w1, w2]
    W = np.ones(X.shape[1], dtype=np.float)

    # Perceptron learning algorithm
    converged = False

    k = 0
    while k < max_itr and not converged:
        # if k % 1000 == 0:
        #     print("Weights at {}-th update is {}".format(k, W))
        Z = np.multiply(np.dot(X, W), Y)
        converged = True
        for j in np.argwhere(Z <= 0).reshape(-1):
            W = W + Y[j] * X[j]
            k += 1
            converged = False

    print("Weights at convergence is {}".format(W))

    # finding accuracy on test data
    Z = np.multiply(np.dot(X_test, W), Y_test)
    num_misclassified = np.argwhere(Z <= 0).reshape(-1).shape[0]
    accuracy = round(100 * (1 - num_misclassified/Y_test.shape[0]), 2)

    print("Count: {}\nMisclassified: {}\nAccuracy: {}%"
          .format(Y_test.shape[0], num_misclassified, accuracy))

    # Saving the results
    x_coord = [np.min(X[:, 1]), np.max(X[:, 1])]
    y_coord = [get_y(x, W) for x in x_coord]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(20, 10)
    # plotting training data
    ax[0].set_title('train')
    ax[0].plot(x_coord, y_coord)
    for x, y in zip(tr_data, tr_labels):
        ax[0].scatter(x[0], x[1], c=color[y], marker=marker[y])

    # plotting test data
    ax[1].set_title('test')
    ax[1].plot(x_coord, y_coord)
    for x, y in zip(ts_data, ts_labels):
        ax[1].scatter(x[0], x[1], c=color[y], marker=marker[y])

    ax[1].text(0, 1, "Count: {}, Missclassified: {}, Accuracy: {}"
               .format(Y_test.shape[0], num_misclassified, accuracy),
               bbox=dict(facecolor='red', alpha=0.5),
               transform=ax[1].transAxes)

    plt.savefig(output_dir + "/" + "result{}.png".format(i))
    plt.clf()
    plt.cla()
    plt.close()

sys.exit(0)
