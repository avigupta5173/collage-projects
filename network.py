import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
class Nurnet(object):
    
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(x,1) for x in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    
    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))
    def update_mini_batch(self, mini_batch, eta):
        nabla_b=0
        nabla_w=0
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b=nabla_b+np.array(delta_nabla_b)
            nabla_w=nabla_w+np.array(delta_nabla_w)
        self.weights = self.weights-(eta/len(mini_batch))*nabla_w
        self.biases = self.biases-(eta/len(mini_batch))*nabla_b  
    def backprop(self, x, y):
        delcbydelw=[]
        delcbydelb=[]
        acurr=x
        actv=[np.array(x)]
        z=[]
        for b,w in zip(self.biases,self.weights):
             zcurr=np.dot(w,acurr)+b
             acurr=sigmoid(zcurr)
             actv.append(acurr)
             z.append(zcurr)
        delcbydelzcur=(actv[-1]-y)*sigmoid_prime(z[-1])
        for a,zv,w in zip(reversed(actv[:-1]),reversed(z[:-1]),reversed(self.weights)):
             delcbydelb.append(delcbydelzcur)
             delcbydelw.append(np.dot(delcbydelzcur,a.transpose()))
             delcbydelzcur=np.dot(w.transpose(),delcbydelzcur)*sigmoid_prime(zv)
        delcbydelb.append(delcbydelzcur)
        delcbydelw.append(np.dot(delcbydelzcur,np.array(x).transpose()))
        return(list(reversed(delcbydelb)),list(reversed(delcbydelw)))
    def evaluate(self,test_data):
        test_result=[(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_result)
    
           
          
       
        
        
    
            
    