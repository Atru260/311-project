from utils import *

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, s):
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
        
        # Average theta_i_k over k subjects for this question
        theta_i_j_avg = 0
        for k in range(s.shape[1]):
            theta_i_j_avg += s[questions[i], k] * theta[users[i], k]
        # Divide by number of subjects in that question
        theta_i_j_avg /= np.sum(s[questions[i]])

        beta_j = beta[questions[i]]
        c_ij = is_correct[i]
        
        log_lklihood += c_ij * (theta_i_j_avg - beta_j)\
                    - scipy.special.logsumexp([0, theta_i_j_avg - beta_j])
                                             
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, s):
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
        
         # Average theta_i_k over k subjects for this question
        theta_i_j_avg = 0
        for k in range(s.shape[1]):
            theta_i_j_avg += s[questions[i], k] * theta[users[i], k]
        # Divide by number of subjects in that question
        theta_i_j_avg /= np.sum(s[questions[i]])
        
        
        
        c_ij = is_correct[i]
        
        beta_j = beta[beta_idx]
        
        # Sigmoid component of derivative
        sig = sigmoid(theta_i_j_avg - beta_j) 
        
        # Calculate derivative for each theta_i_k
        for k in range(s.shape[1]):
            
            s_jk = s[questions[i], k]
            theta_i_k = theta[theta_idx, k]
            
            num_sub_j = np.sum(s[questions[i]])
            
            theta_deriv[theta_idx, k] += (s_jk * theta_i_k / num_sub_j) \
                                       * (c_ij - sig)


        
        beta_deriv[beta_idx] += -c_ij + sig 
        
    # Gradient ascent maximizes log-likelihood
    theta += lr * theta_deriv
    beta += lr * beta_deriv
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, test_data, lr, iterations, num_users, num_questions, k, question_meta):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param k: int, number of subjects to train with
    :param question_meta: A dictionary {question_id: list of subjects for that question}
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # theta[i, k] = theta for student i and subject k
    theta = np.random.rand(num_users, k + 1)
    beta = np.random.rand(num_questions)

    top_k = get_top_k_subjects(question_meta, k)

    s, top_k_enum = generate_subject_matrix(question_meta, top_k)


    val_acc_lst = []
    test_acc_lst = []

    nllk_train = []
    nllk_val = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta, s=s)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta, s=s)
        
        nllk_train.append(neg_lld_train)
        nllk_val.append(neg_lld_val)

        score_val = evaluate(data=val_data, theta=theta, beta=beta, s=s)
        val_acc_lst.append(score_val)
        
        score_test = evaluate(data=test_data, theta=theta, beta=beta, s=s)
        test_acc_lst.append(score_test)
        
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score_val))
        theta, beta = update_theta_beta(data, lr, theta, beta, s)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, test_acc_lst, (nllk_train, nllk_val)


def evaluate(data, theta, beta, s):
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
        
        theta_i_j_avg = np.sum(theta[u] * s[q]) / np.sum(s[q])
        
        x = (theta_i_j_avg - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


# TODO: Move to utils if allowed
def load_question_metadata(root_dir='/data'):
    """ Load the question metadata as a dictionary.

    :param root_dir: str
    :return: A dictionary {question_id: subjects}
   
    """
    path = os.path.join(root_dir, "question_meta.csv")
    
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                q_id = int(row[0])
                subjects = row[1].replace('[', '').replace(']', '').split(sep=',')
                
                for i in range(len(subjects)):
                    subjects[i] = int(subjects[i])
                
                subjects.remove(0)
                data[q_id] = subjects
            except ValueError:
                print('ValueError')
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                print('IndexError')

                pass
    return data

def get_top_k_subjects(meta, k=5):
    '''

    Return the top k most frequently appearing subjects in meta.

    '''
    subjects_freq = {}

    for q in meta:
        subjects = meta[q]
       
        for s in subjects:
            if s in subjects_freq:
                subjects_freq[s] += 1
            else:
                subjects_freq[s] = 1

    top_k = []
    for i in range(k):
        
        m = max(subjects_freq, key=subjects_freq.get)
        top_k.append(m)
        subjects_freq.pop(m)

    return top_k

def generate_subject_matrix(meta, top_k):
    '''

    Create Q x K+1 matrix M, where M[i, j] = 1 if question i comprises subject j
    M[i, j] = 0 otherwise. M[i, 0] = 1 for convenience.


    :param subjects: Dictionary of Q questions: subjects pertaining to that 
        question.
    :param top_k: List of top_k subjects
    :return: A tuple (M, top_k_enum), where top_k_enum maps each subject in top_k to an integer from 1 to K
    '''
    
    q = len(meta)
    k = len(top_k)
    
    subjects = {}
    
    # Filter metadata so that it contains only the top k subjects
    for question in meta:
        
        subjects[question] = []
        
        for subject in meta[question]:
            # Only add top k subjects to dictionary
            if subject in top_k:
                subjects[question].append(subject)
                
                
    # Enumerate the top k subjects
    top_k_enum = {}
    for i in range(len(top_k)):
        top_k_enum[top_k[i]] = i + 1
    
    
    mat = np.zeros((q, k + 1))
    
    for question in subjects:
        mat[question, 0] = k
        for sub in subjects[question]:
            sub_idx = top_k_enum[sub]
            mat[question, sub_idx] = 1
        
    
    return mat, top_k_enum


def main():
    
    # Idea: Students have a different theta for each subject
    
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    num_users = sparse_matrix.shape[0]
    num_questions = sparse_matrix.shape[1]
    
    meta = load_question_metadata('../data')
    
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
   
    
    np.random.seed(100) 
    
    users = train_data['user_id']
    #questions = train_data['question_id']
    #is_correct = train_data['is_correct']
   
    k = 10
    top_k = get_top_k_subjects(meta, k)
    print(top_k)
        
    s, top_k_enum = generate_subject_matrix(meta, top_k)
    print(s)
    print(top_k_enum)
    print(meta[0])
    
    
    
    
   
    lr = 0.02
    iterations = 10
    
    print("Running New IRT with learning rate {} for {} iterations using top {} subjects.".format(lr, iterations, k))

    theta, beta, val_acc_lst, test_acc_lst, nllk \
    = irt(train_data, val_data, test_data, lr, iterations, num_users, num_questions, k, meta)
    
    print("Final validation accuracy:", val_acc_lst[-1])
    print("Final test accuracy:", test_acc_lst[-1])
    
    
    # # Plot average negative log-likelihood for each iteration on both datasets
    # train_nllk, val_nllk = nllk
    
    
    # num_train = len(train_data['user_id'])
    
    # num_val = len(val_data['user_id'])
    
    # # Divide each likelihood value by number of samples to get average
    # train_nllk = list(map(lambda x: x/num_train, train_nllk))
    # val_nllk = list(map(lambda x: x/num_val, val_nllk))
    


    # plt.figure()
    
    # plt.title("Iteration Number vs. Average Negative log-likelihood", y=1.05)
    
    # plt.xlabel('Iteration')
    # plt.ylabel('Average Negative log-likelihood')

    # plt.xticks(range(0, iterations))
    
    # plt.plot(train_nllk, 'g.-', label="Training set")
    # plt.plot(val_nllk, 'b.-', label="Validation set")
    # plt.legend()
    
    # plt.show()
    
    # #####################################################################
    # #                       END OF YOUR CODE                            #
    # #####################################################################

    # #####################################################################
    # # TODO:                                                             #
    # # Implement part (c)                                                #
    # #####################################################################
    # print("Final validation accuracy:", val_acc_lst[-1])
    # print("Final test accuracy:", test_acc_lst[-1])
    
    # # d) Plot probabilities of a correct answer for 5 questions
    
    # # Pick 5 questions, with evenly spaced betas
    
    # num_betas = beta.shape[0]
    
    # beta_idxs = np.array([0, num_betas/4, 2*num_betas/4, 3*num_betas/4, num_betas - 1], dtype=int)
    
    # # Pick questions from sorted betas
    # betas = np.sort(beta)[beta_idxs]
    
    # # Get indices/question ids of sampled betas
    # questions = []
    # for b in betas:
    #     questions.append(np.nonzero(beta==b)[0][0])
    
    # questions = np.array(questions)
    
    # print("Sampled questions", questions)
    
    # print("Betas:", betas)
    
    # # p(c_ij|theta, beta)
    # prob_correct = lambda theta, beta: sigmoid(theta - beta)
    
    # # range of theta values to plot 
    # theta_range = np.arange(start=-5, stop=5, step=.01)
    
    # # p(c_ij|theta, beta) for each question, over theta_range
    # question_probabilities = []
        
    # for beta_j in betas:
    #     question_probabilities.append(prob_correct(theta_range, beta_j))
    
    # # Plot theta vs. probability correct, for each question
    # plt.figure()
    
    # plt.title("Theta Value vs. Probability of answering question correctly", y=1.05)
    
    # plt.xlabel('Theta')
    # plt.ylabel('Probability Correct')
    # #plt.xticks(range(-5,5))
    
    # color_val = 0
    # for i in range(len(question_probabilities)):
        
    #     p = question_probabilities[i]
    #     label = "{},{:.2f}".format(questions[i], betas[i])
        
    #     plt.plot(theta_range, p, '-', label=label, color=hsv_to_rgb((color_val, 1, .7)))
        
    #     color_val += .2
        
    
    # plt.legend(title="Question Number,Beta")
    
    # plt.show()
    
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()