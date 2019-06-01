from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.model = FullyConnectedNet([hidden_dim], input_dim, num_classes, reg=reg, 
                                       weight_scale=weight_scale, dtype=np.float64)
        self.params = self.model.params
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return self.model.loss(X, y)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        all_dims = [input_dim] + hidden_dims + [num_classes]
        
        for idx in range(self.num_layers):
            nrows = all_dims[idx]
            ncols = all_dims[idx+1]
            
            layer_name = "%d" % (idx+1)
            weight_name = "W" + layer_name
            bias_name = "b" + layer_name
            self.params[weight_name] = weight_scale * np.random.randn(nrows, ncols)
            self.params[bias_name] = np.zeros(ncols)
            
            if self.normalization=='batchnorm' and idx < (self.num_layers - 1):
                self.params["gamma" + layer_name] = np.ones(ncols)
                self.params["beta" + layer_name] = np.zeros(ncols)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        scores = X
        caches = {}
        for i in range(1, self.num_layers + 1):
            layer_name = "%d" % i
            W_name = "W" + layer_name
            b_name = "b" + layer_name
            gamma_name = "gamma" + layer_name
            beta_name = "beta" + layer_name
            dropout_name = "dropout" + layer_name
            cache_name = "cache" + layer_name
            
            if self.num_layers == i:
                # Final layer is only affine
                scores, cache = affine_forward(scores, self.params[W_name], self.params[b_name])
            else:
                if self.normalization=='batchnorm':
                    scores, cache = affine_batchnorm_relu_forward(scores, self.params[W_name], self.params[b_name],
                                                                  self.params[gamma_name], self.params[beta_name], 
                                                                  self.bn_params[i-1])
                else:
                    scores, cache = affine_relu_forward(scores, self.params[W_name], self.params[b_name])
                    
                if self.use_dropout:
                    scores, caches[dropout_name] = dropout_forward(scores, self.dropout_param)
                
            caches[cache_name] = cache
            

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        loss, der = softmax_loss(scores, y)
        
        for i in range(self.num_layers, 0, -1):
            layer_name = "%d" % i
            W_name = "W" + layer_name
            b_name = "b" + layer_name
            gamma_name = "gamma" + layer_name
            beta_name = "beta" + layer_name
            dropout_name = "dropout" + layer_name
            cache_name = "cache" + layer_name
            
            # L2 regularization loss. (L2 = λ * sum(W**2))
            loss += self.reg * np.sum(self.params[W_name] * self.params[W_name])
            
            if(self.num_layers == i):
                # final layer
                der, grads[W_name], grads[b_name] = affine_backward(der, caches[cache_name])
            else:
                if self.use_dropout:
                    der = dropout_backward(der, caches[dropout_name])
                    
                if self.normalization=='batchnorm':
                    der, grads[W_name], grads[b_name], grads[gamma_name], grads[beta_name] = affine_batchnorm_relu_backward(
             der, caches[cache_name])
                else:
                    der, grads[W_name], grads[b_name] = affine_relu_backward(der, caches[cache_name])
                
            # regularization gradient
            grads[W_name] += self.reg * 2 * self.params[W_name]
                
        
        ### this should have been done with loops ###
        '''
        # Numeric stability
        scores -= np.max(scores) 
        # compute the class probabilities
        # e^(f)
        exp_scores = np.exp(scores)
        # prob = e^(f) /  ∑j(e^(f))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        
        # compute the loss: average cross-entropy loss
        # loss = − log(prob_i)
        correct_logprobs = -np.log(probs[range(N),y])
        loss = np.sum(correct_logprobs)
        
        # Average loss
        loss /= N 
        
        
        # variables W1, W2, W3, b1, b2, b3, layer1, layer2, hidden1, hidden2, scores, loss
        # gradients dw1, dw2, dw3, db1, db2, db3, dlayer2, dlayer1, dhidden1, dhidden2, dscores, dloss
        # dscores will be computed, dloss/dloss...
        # so, find dw1, dw2, dw3, db1, db2, db3, dlayer2, dlayer1, dhidden1, dhidden2

        # compute the gradient on scores
        # General formula ∂L/∂f_j = prob(j)−𝟙(y(i)=j)
        # ∂L/∂f_j = prob(j)
        dscores = probs
        #∂L/∂f_j = −𝟙(y(i)=j)
        dscores[range(N),y] -= 1
        # Average gradients
        dscores /= N 
        
        # backpropate the gradient to the parameters (W1, W2, W3, b1, b2, b3)
        ### scores = W3 * hidden2 + b3
        # dScores/dW3 = d(W3 * hidden2)/dW3 + db2/dW3 = 1
        #1# dScores/dW3 = hidden2
        # dScores/dhidden2 = d(W3 * hidden2)/dhidden2 + db3/dhidden2
        #2# dScores/dhidden2 = W3
        # dScores/db3 = d(W3 * hidden2)/db3 + db3/db3
        #3# dScores/db3 = 1
        
        # dLoss/dW3 = dLoss/dscores * dScores/dW3
        #4# dLoss/dW3 = dscores * hidden2
        grads['W3'] = hidden2.T.dot(dscores)
        # dLoss/dhidden2 = dLoss/dscores * dScores/dhidden2
        #5# dLoss/dhidden2 = dscores * W3
        dhidden2 = dscores.dot(W3.T)
        # dLoss/db3 = dLoss/dscores * dScores/db3
        #6# dLoss/db3 = dscores * 1
        grads['b3'] = dscores.sum(axis=0)
        
        # dLoss/dlayer2 = dLoss/dhidden2 * dhidden2/dlayer2
        ### hidden2 = np.maximum(layer2scores, 0)
        #7# dhidden2/dlayer2 = 1 if layer2scores > 0, 0 otherwise
        dhidden2_dlayer2 = np.zeros_like(layer2scores)
        dhidden2_dlayer2[layer2scores > 0] = 1
        #8# dlayer2 = dhidden2 * dhidden2/dLayer2
        dlayer2 = dhidden2 * dhidden2_dlayer2
        
        ### layer2 = W2 * hidden + b2
        # dlayer2/dW2 = d(W2 * hidden)/dW2 + db2/dW2 
        #9# dlayer2/dW2 = hidden 
        # dlayer2/dhidden = d(W2 * hidden)/dhidden + db2/dhidden 
        #10# dlayer2/dhidden = W2 
        # dlayer2/db2 = d(W2 * hidden)/db2 + db2/db2 
        #11# dlayer2/db2 = 1 
        
        # dLoss/dW2 = dLoss/dlayer2 * dlayer2/dW2
        #12# dLoss/dW2 = dlayer2 * hidden
        grads['W2'] = hidden.T.dot(dlayer2)
        # dLoss/dhidden = dLoss/dlayer2 * dlayer2/dhidden
        #13# dLoss/dhidden = dlayer2 * W2
        dhidden = dlayer2.dot(W2.T)
        # dLoss/db2 = dLoss/dscores * dScores/db2
        #14# dLoss/db2 = dlayer2 * 1
        grads['b2'] = dlayer2.sum(axis=0)
        
        # dLoss/dlayer1 = dLoss/dhidden * dhidden/dlayer1
        ### hidden = np.maximum(layer1scores, 0)
        #15# dhidden/dlayer1 = 1 if layer1scores > 0, 0 otherwise
        dhidden_dlayer1 = np.zeros_like(layer1scores)
        dhidden_dlayer1[layer1scores > 0] = 1
        #16# dlayer1 = dhidden * dhidden/dLayer1Scores
        dlayer1 = dhidden * dhidden_dlayer1
        
        ### layer1 = W1 * X + b1
        # dlayer1/dW1 = d(W1 * X)/dW1 + db1/dW1 
        #17# dlayer1/dW1 = X 
        # dlayer1/db1 = d(W1 * X)/db1 + db1/db1 
        #18# dlayer1/db1 = 1 
        
        # dLoss/dW1 = dLoss/dlayer1 * dlayer1/dW1
        #19# dW1 = dlayer1 * X
        grads['W1'] = X.T.dot(dlayer1)
        # dLoss/db1 = dLoss/dlayer1 * dlayer1/db1
        #20# dLoss/db1 = dlayer1 * 1
        grads['b1'] = dlayer1.sum(axis=0)
        
        
        grads['W1'] = np.reshape(grads['W1'], (D, -1))
        
        # regularization gradient 
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        '''

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
