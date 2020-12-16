from utils import *
import part_b.item_response_2 as irt2

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb
def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    
    users = data['user_id']
    questions = data['question_id']
    is_correct = data['is_correct']
    
    for i in range(len(users)):
        theta_i = theta[users[i]]
        beta_j = beta[questions[i]]
        c_ij = is_correct[i]
        log_lklihood += c_ij * (theta_i - beta_j)\
                    - scipy.special.logsumexp([0, theta_i - beta_j])
                                             
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    # num_students = data.shape[0]
    # num_questions = data.shape[1]   
    
    users = data['user_id']
    questions = data['question_id']
    is_correct = data['is_correct']
    
    theta_deriv = np.zeros(theta.shape)
    beta_deriv = np.zeros(beta.shape)
    
    for i in range(len(users)):
        
        theta_idx = users[i]
        beta_idx = questions[i]
        
        theta_i = theta[theta_idx]
        beta_j = beta[beta_idx]
        c_ij = is_correct[i]
        
        sig = sigmoid(theta_i - beta_j) 
        theta_deriv[theta_idx] += c_ij - sig
        beta_deriv[beta_idx] += -c_ij + sig 
        
    # Gradient ascent maximizes log-likelihood
    theta += lr * theta_deriv
    beta += lr * beta_deriv
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, test_data, lr, iterations, num_users, num_questions):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: The sparse matrix (num_users * num_questions)
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(num_users)
    beta = np.random.rand(num_questions)

    val_acc_lst = []
    test_acc_lst = []
    train_acc_lst = []
    
    nllk_train = []
    nllk_val = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        
        nllk_train.append(neg_lld_train)
        nllk_val.append(neg_lld_val)

        
        
        score_val = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score_val)
        
        score_test = evaluate(data=test_data, theta=theta, beta=beta)
        test_acc_lst.append(score_test)
        
        score_train = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(score_train)
        
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score_val))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, test_acc_lst, train_acc_lst, (nllk_train, nllk_val)

def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])
           
           
def irt_sparse(train_sparse, v_or_t, lr, iterations, num_users, num_questions):
    ''' 
    Run IRT prediction using the sparse matrix representation of the training data
    '''
    train_dict = {'user_id':[], 'question_id':[], 'is_correct':[]}
    
    # Convert sparse matrix to dictionary before feeding it to the algo.
    for user_id in range(train_sparse.shape[0]):
        for question_id in range(train_sparse.shape[1]):
            is_correct = train_sparse[user_id, question_id]
            if not np.isnan(is_correct):
                train_dict['user_id'].append(user_id)
                train_dict['question_id'].append(question_id)
                train_dict['is_correct'].append(is_correct)
    predictions = irt_p(train_dict, v_or_t, lr, iterations, num_users, num_questions)
    predictions = np.where(predictions == True, 1, predictions)
    predictions = np.where(predictions == False, 0, predictions)
    return predictions
    



           
def irt_p(data, v_or_t, lr, iterations, num_users, num_questions):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: The sparse matrix (num_users * num_questions)
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(num_users)
    beta = np.random.rand(num_questions)

    # val_acc_lst = []
    # test_acc_lst = []

    # nllk_train = []
    # nllk_val = []

    for i in range(iterations):
        # neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        # neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        
        # nllk_train.append(neg_lld_train)
        # nllk_val.append(neg_lld_val)

        
        
        # score_val = evaluate(data=val_data, theta=theta, beta=beta)
        # val_acc_lst.append(score_val)
        
        # score_test = evaluate(data=test_data, theta=theta, beta=beta)
        # test_acc_lst.append(score_test)
        
        # print("NLLK: {} \t Score: {}".format(neg_lld_train, score_val))
        p = predict(v_or_t, theta, beta)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return p

def predict(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        #pred.append(p_a >= 0.5)
        pred.append(p_a)
    return np.array(pred)

def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    num_users = sparse_matrix.shape[0]
    num_questions = sparse_matrix.shape[1]
    
    
    
    
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
   
    
    np.random.seed(100) 
   
    lr = 0.02
    iterations = 10
    
    print("Running IRT with learning rate {} for {} iterations.".format(lr, iterations))

    theta, beta, val_acc_lst, test_acc_lst, train_acc_lst, nllk \
        = irt(train_data, val_data, test_data, lr, iterations, num_users, num_questions)
    
    
    
    
    # Plot average negative log-likelihood for each iteration on both datasets
    train_nllk, val_nllk = nllk
    
    
    num_train = len(train_data['user_id'])
    
    num_val = len(val_data['user_id'])
    
    # Divide each likelihood value by number of samples to get average
    train_nllk = list(map(lambda x: x/num_train, train_nllk))
    val_nllk = list(map(lambda x: x/num_val, val_nllk))
    


    plt.figure()
    
    plt.title("Iteration Number vs. Average Negative log-likelihood", y=1.05)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Negative log-likelihood')

    plt.xticks(range(0, iterations))
    
    plt.plot(train_nllk, 'g.-', label="Training set")
    plt.plot(val_nllk, 'b.-', label="Validation set")
    plt.legend()
    
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    print("Final validation accuracy:", val_acc_lst[-1])
    print("Final test accuracy:", test_acc_lst[-1])
    print("Final training accuracy", train_acc_lst[-1])
    
    # d) Plot probabilities of a correct answer for 5 questions
    
    # Pick 5 questions, with evenly spaced betas
    
    num_betas = beta.shape[0]
    
    beta_idxs = np.array([0, num_betas/4, 2*num_betas/4, 3*num_betas/4, num_betas - 1], dtype=int)
    
    # Pick questions from sorted betas
    betas = np.sort(beta)[beta_idxs]
    
    # Get indices/question ids of sampled betas
    questions = []
    for b in betas:
        questions.append(np.nonzero(beta==b)[0][0])
    
    questions = np.array(questions)
    
    print("Sampled questions", questions)
    
    print("Betas:", betas)
    
    # p(c_ij|theta, beta)
    prob_correct = lambda theta, beta: sigmoid(theta - beta)
    
    # range of theta values to plot 
    theta_range = np.arange(start=-5, stop=5, step=.01)
    
    # p(c_ij|theta, beta) for each question, over theta_range
    question_probabilities = []
        
    for beta_j in betas:
        question_probabilities.append(prob_correct(theta_range, beta_j))
    
    # Plot theta vs. probability correct, for each question
    plt.figure()
    
    plt.title("Theta Value vs. Probability of answering question correctly", y=1.05)
    
    plt.xlabel('Theta')
    plt.ylabel('Probability Correct')
    
    color_val = 0
    for i in range(len(question_probabilities)):
        
        p = question_probabilities[i]
        label = "{} ({:.2f})".format(questions[i], betas[i])
        
        plt.plot(theta_range, p, '-', label=label, color=hsv_to_rgb((color_val, 1, .7)))
        
        color_val += .2
        
    
    plt.legend(title="Question Number (Beta)")
    
    plt.show()
    
    
    
    # Plot the accuracies with item_response_2
    train_nllk_2, val_nllk_2, \
        train_acc_lst_2, val_acc_lst_2, test_acc_lst_2 = irt2.main()
    
    plt.figure()
    
    plt.title("Comparison of augmented IRT model with baseline accuracies", y=1.05)
    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    
    train_col = hsv_to_rgb((.1, 1, .75))
    val_col = hsv_to_rgb((.7, 1, .7))
    test_col= hsv_to_rgb((.45, 1, .8))
        
    plt.plot(train_acc_lst, '-', label="train", color=train_col)
    plt.plot(train_acc_lst_2, '--', label="train 2", color=train_col)
    plt.plot(val_acc_lst, '-', label="validation", color=val_col)
    plt.plot(val_acc_lst_2, '--', label="validation 2", color=val_col)
    plt.plot(test_acc_lst, '-', label="test", color=test_col)
    plt.plot(test_acc_lst_2, '--', label="test 2", color=test_col)        
    
    plt.legend()
    
    plt.show()
    
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
