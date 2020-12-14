# TODO: complete this file.
import os
#print(os.getcwd())
os.chdir("..")
print(os.getcwd())

from utils import *
from part_a.knn import knn_impute_by_user_p
from part_a.knn import knn_impute_by_user
from part_a.neural_network import AutoEncoder
from part_a.neural_network import train_p
from part_a.neural_network import predict
from part_a.item_response import irt_sparse
import torch

# from https://towardsdatascience.com/you-should-care-about-bootstrapping-ced0ffff2434

def bootstrap(data,n_trials):
    np.random.seed(100) 
    index = np.arange(data.shape[0])
    bootstrap_index = np.random.choice(index,
                                       size=data.shape[0]*n_trials,
                                       replace=True)
    bootstrap_data = np.reshape(data[bootstrap_index,:],
                                (n_trials,*data.shape))
    return bootstrap_data

def model(model_name, bag, v_or_t):
    knn_k = 24
    lr_i = 0.02
    iterations = 10
    num_users = bag.shape[0]
    num_questions = bag.shape[1]
    k = 50
    m = AutoEncoder(num_question=1774, k=k)

    # Set optimization hyperparameters.
    lr_n = 0.001
    num_epoch = 32
    lamb = 0.01
    if model_name == 'k':
        # knnl = []
        # for i in range(1, knn_k):
        #     knn = knn_impute_by_user(bag, v_or_t, i)
        #     knnl.append(knn)
        # knnl = np.array(knnl)
        # best_k = np.argmax(knnl) + 1
        # print('best k =')
        # print(best_k)
        return knn_impute_by_user_p(bag, v_or_t, knn_k)
    if model_name == 'i':
        p_irt = irt_sparse(bag, v_or_t, lr_i, iterations, num_users, num_questions)
        return p_irt
    if model_name == 'n':
        zero_train_matrix = bag.copy()
    # Fill in the missing entries to 0.
        zero_train_matrix[np.isnan(bag)] = 0
    # Change to Float Tensor for PyTorch.
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(bag)
        train_p(m, lr_n, lamb, train_matrix, zero_train_matrix, num_epoch, v_or_t)
        return predict(m, zero_train_matrix, v_or_t)
    
    """ Train more than 2 of the same model"""
def train_e(model_name, bags, start_index, end_index, v_or_t):
    total = model(model_name, bags[start_index], v_or_t)
    for i in range(start_index+1, end_index+1):
        print(i)
        total= total + model(model_name, bags[i], v_or_t)
    return total
def main():
    trials = 50
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    bags = bootstrap(sparse_matrix, trials)
    print(bags[38].shape)
    print(bags[37].shape)
    #p_knn = train_e('k', bags, 0, 47, val_data)
    p_knn = model('k', bags[38], val_data)
    # p_knn2 = model('k', bags[3], val_data)
    # p_knn3 = model('k', bags[4], val_data)
    # p_knn4 = model('k', bags[5], val_data)
    # p_knn5 = model('k', bags[6], val_data)
    # p_knn6 = model('k', bags[7], val_data)
    # p_knn7 = model('k', bags[8], val_data)
    p_irt = model('i', bags[48], val_data)
    print('neural network hyperparameter testing...')
    p_nn = model('n', bags[49], val_data)
    # #print(p_irt)
    prob = p_irt + p_nn
    prob = prob/2
    prob = np.where(prob >= .5, 1, prob)
    prob = np.where(prob <= .5, 0, prob)
    total = p_knn +prob
    total = total/(trials-1)
    predict = evaluate(val_data, total)
    #print(p_nn)
    print(predict)

    


if __name__ == "__main__":
    main()
