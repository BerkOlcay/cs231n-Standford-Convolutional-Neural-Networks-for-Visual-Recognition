from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    for i in range(num_train):
        # f(x_i,W)
        scores = X[i].dot(W) 
        # Numeric stability
        scores -= np.max(scores) 
        correct_class_score = scores[y[i]]
        # ∑j(e^(f_j))
        denominator = np.sum(np.exp(scores)) 
        # prob_i = e^(f_i) /  ∑j(e^(f_j))
        probs = np.exp(correct_class_score) / denominator 
        # L_i = − log(prob_i)
        correct_logprobs = -np.log(probs) 
        loss += correct_logprobs
                
        #∂L/∂f_j = prob(j)−𝟙(y(i)=j)
        for j in range(num_class):
            #∂L/∂f_j = prob(j)
            dW[:, j] += np.exp(scores[j]) / denominator * X[i, :] 
            #(y(i)=j)
            if j == y[i]:
                #∂L/∂f_j = −𝟙(y(i)=j)
                dW[:, j] -= X[i, :] 
       
    # Average loss
    loss /= num_train 
    # Average gradients
    dW /= num_train 
    
    # regularization loss.
    loss += reg * np.sum(W * W) 
    # regularization gradient
    dW += reg * W 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    # evaluate class scores, [N x K]
    # f(x_i,W)
    scores = np.dot(X, W)
    # Numeric stability
    scores -= np.max(scores) 

    # compute the class probabilities
    # e^(f)
    exp_scores = np.exp(scores)
    # prob_i = e^(f_i) /  ∑j(e^(f_j))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # compute the loss: average cross-entropy loss
    # loss = − log(prob_i)
    correct_logprobs = -np.log(probs[range(num_train),y])
    loss = np.sum(correct_logprobs)

    # compute the gradient on scores
    # General formula ∂L/∂f_j = prob(j)−𝟙(y(i)=j)
    # ∂L/∂f_j = prob(j)
    dscores = probs
    #∂L/∂f_j = −𝟙(y(i)=j)
    dscores[range(num_train),y] -= 1

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)

    
    # Average loss
    loss /= num_train 
    # Average gradients
    dW /= num_train 
    
    # regularization loss.
    loss += reg * np.sum(W * W) 
    # regularization gradient
    dW += reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
