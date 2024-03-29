# import os
# #print(os.getcwd())
# os.chdir("..")

from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances

from utils import *
import numpy as np
import matplotlib.pyplot as plt


def weighted_nan_euclidean(X, Y=None, squared=False, missing_values=np.nan, copy=True):
    return nan_euclidean_distances(X, Y, squared, missing_values, copy)

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc

def knn_impute_by_user_p(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
  
    
    
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {} k: {}".format(acc, k))
    prediction = sparse_matrix_predictions(valid_data, mat)
    return np.array(prediction)



def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    acc_user = []
    acc_item = []
    K = [1, 6, 11, 16, 21, 26]
    for k in K:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_user.append(acc)
    print(acc_user)
    np.array(acc_user)
    i = np.argmax(acc_user)
    tacc_user = knn_impute_by_user(sparse_matrix, test_data, K[i])
    print(tacc_user)
    for k in K:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_item.append(acc)
    print(acc_item)
    np.array(acc_item)
    i = np.argmax(acc_item)
    tacc_user = knn_impute_by_item(sparse_matrix, test_data, K[i])
    print(tacc_user)
    K = np.array(K)
    acc_user = np.array(acc_user)
    acc_item = np.array(acc_item)
    plt.plot(K, acc_user, 'r') # plotting t, a separately 
    plt.plot(K, acc_item, 'b') # plotting t, b separately 
    plt.title("Validation Accuracy over k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
