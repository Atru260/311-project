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
from scipy import sparse
import torch

# from https://towardsdatascience.com/you-should-care-about-bootstrapping-ced0ffff2434

def bootstrap(data,n_trials, sparse_shape):
    np.random.seed(100) 
    
    # Arrange training data into a numpy array with columns user_id, question_id, and is_correct
    train = np.array([np.array(x) for x in [data['user_id'], data['question_id'], data['is_correct']]]).T
    
    index = np.arange(train.shape[0])

    
    bootstrap_index = np.random.choice(index,
                                       size=train.shape[0]*n_trials,
                                       replace=True)
    bootstrap_data = np.reshape(train[bootstrap_index,:],
                                (n_trials,*train.shape))
    
    # Convert to sparse matrix
    spdata = np.full((n_trials, *sparse_shape), np.nan)
    
    for t in range(n_trials):
        
        trial_boots = bootstrap_data[t].T
        
        # Set user_id, question_id index to is_correct
        spdata[t][trial_boots[0], trial_boots[1]] = trial_boots[2]
        #spdata[t] = sparse.csr_matrix(spdata[t])
    return spdata
    

def model(model_name, bag, v_or_t):
    knn_k = 9
    lr_i = 0.02
    iterations = 10
    num_users = bag.shape[0]
    num_questions = bag.shape[1]
    k = 50
    m = AutoEncoder(num_question=1774, k=k)

    # Set optimization hyperparameters.
    lr_n = 0.05
    num_epoch = 11
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
    trials = 3
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")
    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    
    train_data = load_train_csv("data")

    bags = bootstrap(train_data, trials, sparse_matrix.shape)
    print(bags.shape)
    p_irtv = train_e('i', bags, 0, 2, val_data)
    p_irtv = p_irtv/3
    p_irtv = np.where(p_irtv >= .5, 1, p_irtv)
    p_irtv = np.where(p_irtv <= .5, 0, p_irtv)
    val = evaluate(val_data, p_irtv)
    print('Validation Accuracy:')
    print(val)
    p_irt = train_e('i', bags, 0, 2, test_data)
    prob = p_irt 
    prob = prob/3
    prob = np.where(prob >= .5, 1, prob)
    prob = np.where(prob <= .5, 0, prob)
    predict = evaluate(test_data, prob)
    print('Test Accuracy:')
    print(predict)

    


if __name__ == "__main__":
    main()
