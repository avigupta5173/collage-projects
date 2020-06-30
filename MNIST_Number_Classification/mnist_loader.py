import _pickle as cPickle
import gzip
import numpy as np

def load():
    load=gzip.open('mnist.pkl.gz','rb')
    training_data,validation_data,test_data=cPickle.load(load,encoding='latin1')
    load.close()
    training_input=[np.reshape(x,(784,1)) for x in training_data[0]]
    training_result=[vectorize(y) for y in training_data[1]]
    training_data=list(zip(training_input,training_result))
    validation_inputs=[np.reshape(x,(784,1)) for x in validation_data[0]]
    validation_data=list(zip(validation_inputs,validation_data[1]))
    test_input=[np.reshape(x,(784,1)) for x in test_data[0]]
    test_data=list(zip(test_input,test_data[1]))
    return(training_data,validation_data,test_data)

def vectorize(a):
    e=np.zeros((10,1))
    e[a]=1
    return e
    
