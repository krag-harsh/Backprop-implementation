import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

class MyNeuralNetwork():
  """
  My implementation of a Neural Network Classifier.
  """
  acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
  weight_inits = ['zero', 'random', 'normal']

  def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
    if activation not in self.acti_fns:
        raise Exception('Incorrect Activation Function')
    if weight_init not in self.weight_inits:
        raise Exception('Incorrect Weight Initialization Function')

    self.n_layers=n_layers #int value specifying the number of layers
    self.layer_sizes=layer_sizes #integer array of size n_layers specifying the number of nodes in each layer
    self.activation=activation  # string specifying the activation function to be used, possible inputs: relu, sigmoid, linear, tanh
    self.learning_rate=learning_rate #float value specifying the learning rate to be used
    self.weight_init=weight_init #string specifying the weight initialization function to be used,possible inputs: zero, random, normal
    self.batch_size=batch_size #int value specifying the batch size to be used
    self.num_epochs=num_epochs #int value specifying the number of epochs to be used

    self.W={}  ## this will contain all the parameters
    self.B={}

    self.Zs={}  ## this will contain values after forward prop useful for back prop
    self.As={}
    self.initialize_weights_byinput()  # initializing the weights
    ## specifing the funciton according to parameter for future use
    if activation=="relu":
      self.a_fn = self.relu
      self.a_fng = self.relu_grad
    elif activation=="sigmoid":
      self.a_fn = self.sigmoid
      self.a_fng = self.sigmoid_grad
    elif activation=="linear":
      self.a_fn = self.linear
      self.a_fng = self.linear_grad
    elif activation=="tanh":
      self.a_fn = self.tanh
      self.a_fng = self.tanh_grad
    self.train_loss=[]  # this will store loss during training
    self.val_loss=[]

  def initialize_weights_byinput(self):
    wtype=self.weight_init
    if(wtype=="zero"):
      for i in range(1,self.n_layers):
        self.W[i]=self.zero_init((self.layer_sizes[i],self.layer_sizes[i-1])) ### shape of each W matrix will be #of neurons in that layerXthat in previous layer
        self.B[i]=self.zero_init((self.layer_sizes[i],1))  
    elif(wtype=="random"):
      for i in range(1,self.n_layers):
        self.W[i]=self.random_init((self.layer_sizes[i],self.layer_sizes[i-1]))
        self.B[i]=self.random_init((self.layer_sizes[i],1))
    else:
      for i in range(1,self.n_layers):
        self.W[i]=self.normal_init((self.layer_sizes[i],self.layer_sizes[i-1]))
        self.B[i]=self.normal_init((self.layer_sizes[i],1))

  def compare_with_mlp(self,loss_from_your_model,X,y):
    # comparing with MLP
    # X is all data and Y is all labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation ='relu',solver = 'sgd' , alpha = 0, batch_size = 32,max_iter=100, learning_rate_init = 0.001, learning_rate = 'constant',shuffle=True,momentum = 0,nesterovs_momentum=False,validation_fraction = 0.1)
    mlp.fit(X_train, y_train)
    loss_from_sklearn = mlp.loss_curve_

    plt.plot(loss_from_sklearn,label="sklearn")
    plt.plot(loss_from_your_model,label="your NN")
    plt.legend(loc="upper left")
    plt.savefig("result.png")
    plt.close()

  def relu(self, X):
    return np.maximum(0,X)

  def relu_grad(self, X):
    f=X
    f[X<0]=0
    f[X>0]=1.0
    return f

  def sigmoid(self, X):
    si= 1+np.exp(-X)
    si=1/si
    return si

  def sigmoid_grad(self, X):
    s=self.sigmoid(X) 
    return s*(1-s)

  def linear(self, X):
    return X

  def linear_grad(self, X):
    s=X.shape
    return np.ones(s)

  def tanh(self, X):
    return np.tanh(X)

  def tanh_grad(self, X):
    return 1-(self.tanh(X)**2)

  def softmax(self, X):
    e=np.exp(X)
    de=np.sum(e,axis=0,keepdims=True)
    return e/de

  def softmax_grad(self, X):
    s=self.softmax(X)
    jm=np.diag(s)  # gradient of a softmax is a 2d matrix(jacobian)
    l=len(jm)
    for i in range(l):
      for j in range(l):
        if(i!=j):
          jm[i][j]=-s[i]*s[j]
        else:
          jm[i][j]=s[i]*(1-s[i])
    return jm

  def zero_init(self, shape):
    return np.zeros(shape)

  def random_init(self, shape):
    return np.random.randn(shape[0],shape[1])*0.01

  def normal_init(self, shape):
    return np.random.normal(size=shape)*0.01

  def fit(self, X, y, xval,yval):
    # fit function has to return an instance of itself or else it won't work with test.py
    m=len(X)
    bs=self.batch_size
    a=[i for i in range(m)]  # storing the indexes
    # toseloss=[]
    for e in range(self.num_epochs):
      np.random.shuffle(a)  # suffled the training indexes
      losses=0  # for one batch initilising loss as 0
      for i in range(0,m//bs):
        ind=a[bs*i:bs*(i+1)]  # getting the indexes to use
        X_t=X[ind]  # batch_sizex784
        Y_t=y[ind]
        output=self.predict_proba(X_t)  # finding the probability on the input
        loss=self.crossentropyloss(output,Y_t)  # getting the loss
        self.backpropagation(loss, Y_t)  # backpropagating and updating weights
        # print(f"After epoch {e+1} loss at i {i} = {loss}")
        losses+=loss
      if((e+1)%5==0):
        print(f"Epoch {e+1} done")
      self.train_loss.append(losses/(m//bs))  # saving the losses
      self.validate(xval,yval)  # saving validation loss
      if((e+1)%50==0):  # after every 50 epochs saving the weights and biases
        print("epoch 50,saving model parameters")
        p_namew=f"{e+1}_{self.activation}_weights.pkl"
        p_nameb=f"{e+1}_{self.activation}_biases.pkl"
        f1=open(p_namew, "wb")
        pickle.dump(self.W,f1)  # saving weights
        f1.close()
        f2=open(p_nameb, "wb")
        pickle.dump(self.B,f2)  # saving baises
        f2.close()
        print(f"After epoch {e+1} Train loss = {self.train_loss[-1]}")
    return self

  def validate(self,xval,yval):  # wrote this function for validation
    output=self.predict_proba(xval)
    loss=self.crossentropyloss(output,yval)
    self.val_loss.append(loss)

  def predict_proba(self, X):
    """
    Predicting probabilities using the trained linear model.
    Parameters
    X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.
    Returns
    y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
        class wise prediction probabilities.
    """
    # return the numpy array y which contains the predicted values
    a=X.T  # make a of shape 728xm
    self.Zs={}
    self.As={}
    self.As[0]=a # first layer activation is same as input
    for i in range(1,self.n_layers-1):
      z=self.W[i].dot(a)+self.B[i]  # (ix(i-1)) X ((i-1)xm) = ixm
      self.Zs[i]=z
      a=self.a_fn(z)  # applying activation function # ixm
      self.As[i]=a  
    z=self.W[self.n_layers-1].dot(a)
    self.Zs[self.n_layers-1]=z
    a=self.softmax(z)  # applying softmax in the last layer
    self.As[self.n_layers-1]=a
    return a.T

  def predict(self, X):
    pre_prob=self.predict_proba(X)
    return np.argmax(pre_prob.T,axis=0)  # finding the index of the maximum probability

  def score(self, X, y):
    yp=self.predict(X)  # getting the probabilities
    a= (yp==y)
    a=a.sum()
    return a/len(y)

  def crossentropyloss(self, ypred, ytrue):
    one_hot_targets = np.eye(10)[ytrue]   # 10 is the final number of class
    a=one_hot_targets*np.log(ypred + 1e-9) 
    a= -np.sum(a)
    return a/len(ytrue)

  def backpropagation(self,loss,ytrue):  # creating this function which will update the parameters
    nl=self.n_layers
    m=ytrue.shape[0]
    one_hot_targets = np.eye(10)[ytrue]  # (m, 10)
    error= self.As[nl-1]-one_hot_targets.T  # (10,m)
    dz=error
    # from da[l] have to find da[l-1], dw[l], db[l]
    dw= dz.dot(self.As[nl-2].T)/m   # (ixm) X ((i-1)xm)T = (ix(i-1))
    db=np.sum(dz,keepdims=True, axis=1)/m
    da_l_minus_one= np.dot(self.W[nl-1].T, dz)  # (ix(i-1)).T X (ixm) = (i-1)xm
    # update parameter of last layer
    self.W[nl-1]= self.W[nl-1] - self.learning_rate*dw
    self.B[nl-1]=self.B[nl-1] - self.learning_rate*db
    for i in range(nl-2,0,-1):
      dz=da_l_minus_one* self.a_fng(self.Zs[i])  # same as a[l]= ixm
      dw=np.dot(dz, self.As[i-1].T)/m  # (ixm) X ((i-1)xm)T = (ix(i-1))
      db=np.sum(dz,keepdims=True, axis=1)/m
      da_l_minus_one=np.dot(self.W[i].T,dz)   # (ix(i-1)).T X (ixm) = (i-1)xm
      # update parameters
      self.W[i]= self.W[i] - self.learning_rate*dw
      self.B[i]= self.B[i] - self.learning_rate*db
