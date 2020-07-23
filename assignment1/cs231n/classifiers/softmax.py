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
    
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  mask = np.zeros((N,C))
  for i in xrange(N):
    for j in xrange(C):
        mask[i,j] = X[i,:].dot(W[:,j])
    mask[i] -= np.max(mask[i])            # adding C regulizar mask(i,j) = exp(<Xi,Wj>-max_k<Xi,Wk>)
    #mask[i] -= mask[np.argmax(mask[i])]  # adding C regulizar mask(i,j) = exp(<Xi,Wj>-max_k<Xi,Wk>)
    mask[i] = np.exp(mask[i])
    mask[i] /= np.sum(mask[i])
    loss += -np.log(mask[i,y[i]])
    
  loss /= N  
  loss += reg*np.sum(W*W)
  
  for i in range(N):
        for j in range(C):
            dW[:, j] += mask[i,j] * X[i]
        dW[:,y[i]] -= X[i]
  
  dW /= N
  dW += reg*2*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
    
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
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  mask = np.exp(X.dot(W))
  # mask -= np.max(mask,axis=1)[:, np.newaxis] # issued an error of "divide by zero encountered in log"
  mask = mask / np.sum(mask, axis=1)[:, np.newaxis]
  loss = np.sum(-np.log(mask[np.arange(N),y]))
  loss /= N
  loss += reg*np.sum(W*W)
    
  mask[np.arange(N),y] -= 1
  dW += X.T.dot(mask)
  dW /= N
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

