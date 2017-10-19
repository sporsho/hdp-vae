###class to manipulate cifar-100 images ###
import numpy as np
import urllib
import cPickle
import os.path

class CIFAR_100(object):
    """class for cifar 100"""
    def unpickle(self, filename):
        fo= open(filename, 'rb')
        dictData=cPickle.load(fo)
        fo.close()
        return dictData

    
    def onehot_fine_labels(self, labels):
        return np.eye(100)[labels]
    
    def onehot_coarse_labels(self, labels):
        return np.eye(20)[labels]
    
    def structureImages(self, rawImage):
        raw_float= np.array(rawImage, dtype=float)
        images= raw_float.reshape([-1,3,32,32])
        images=images.transpose([0,2,3,1])
        return images
    
    def __init__(self):
        testUrl="http://kashaaf.com/cifar-100-python/test" 
        trainUrl= "http://kashaaf.com/cifar-100-python/train"
        opener= urllib.URLopener()
        if not os.path.isfile('train'):
            opener.retrieve(trainUrl, 'train')
        if not os.path.isfile('test'):
            opener.retrieve(testUrl, 'test')
        self.x_train=self.unpickle('train')['data']
        self.x_test=self.unpickle('test')['data']
        self.y_fine_train=self.onehot_fine_labels(self.unpickle('train')['fine_labels'])
        self.y_coarse_train=self.onehot_coarse_labels(self.unpickle('train')['coarse_labels'])
        self.y_fine_test= self.onehot_fine_labels(self.unpickle('test')['fine_labels'])
        self.y_coarse_test=self.onehot_coarse_labels(self.unpickle('test')['coarse_labels'])
        self.fine_label_names= self.unpickle('meta')['fine_label_names']
        self.coarse_label_names= self.unpickle('meta')['coarse_label_names']
        

        
