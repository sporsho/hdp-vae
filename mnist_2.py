import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
class MNIST(object):
    def __init__(self):
        mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.x_train=mnist.train.images
        self.x_test=mnist.test.images
        self.train_super=[]
        self.test_super=[]
        self.y_train= mnist.train.labels
        self.y_test=mnist.test.labels
        counter=0
        for y in self.y_train:
            dat= self._one_hot_to_int(y)
            if dat%2==0:
                self.train_super.append([0.,1.])
            else:
                self.train_super.append([1.,0.])
        for y in self.y_test:
            dat= self._one_hot_to_int(y)
            if dat%2==0:
                self.test_super.append([0.,1.])
            else:
                self.test_super.append([1.,0.])
    def _one_hot_to_int(self, arr):
        flag=0
        for i in range(10):
            if (arr[i]==1.):
                flag=1
                return i
        if(flag==0):
            return 0
m= MNIST()