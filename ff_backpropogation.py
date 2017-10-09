import numpy as np

class ANN:        
        
    def __init__(self,learning_rate = 0.1,w=[],b=[]):
        self.learning_rate = learning_rate
        self.set_weights(w,b)
    
    def random_weights(self, shape):
        '''
        input shape includes how many inputs to the net and then how many nodes on each layer. All in a tuple.
        '''
        weights = []
        biases = []
        
        for i in range(1,len(shape)):
            weights.append(np.matrix(np.random.random((shape[i],shape[i-1]))*2.0 - 1.0))
            biases.append(np.matrix(np.random.random(shape[i])*2.0 - 1.0).T)
                          
        self.set_weights(weights,biases)
            
    def set_weights(self,w,b):
        '''
        Takes a three dimensional matrix.
        0 axis is the layers.
        1 axis is the neurons on that layer.
        2 axis is the incoming weights for each neuron. Including the bais at the end of vector(so each array is > 1 
             and n-1 inputs for the neuron given an array of size n).
        
        '''
        
        
        self.weights = w
        self.biases = b
       
    def _tanh(self,n):
        return np.tanh(n)
    
    def _derivative_tanh(self,n):
        return 1.0 - np.power(np.tanh(n),2)
    
    def _logistic_sigmoid(self,n):
        return 1.0/(1.0 + np.power(np.e,-n))
    
    def forward(self, inputs):
        
        output=np.matrix(inputs).T
        for i in range(len(self.weights)):
            #there is one row for each neuron in the weight matrix the statement will produce output for each neuron. 
            output= self._tanh(self.weights[i] * output + self.biases[i])
  
        
        return output
    
    def _derivative_logistic_sigmoid(self,n):
        n = self._logistic_sigmoid(n)
        return np.multiply(n,(1.0 - n))
    
    def  backpropogation(self,inputs, expected):
        outputs=[np.matrix(inputs).T]
        for i in range(len(self.weights)):
            #there is one row for each neuron in the weight matrix the statement will produce output for each neuron. 
            outputs.append(self._tanh(self.weights[i] * outputs[i] + self.biases[i]))
  
            
        output_error = 0.0
        error = 0.0
        #recall one more output element than weight elements
        for i in range(len(outputs)-1,0,-1):
            if i == len(outputs) - 1:
                error = outputs[i] - np.matrix(expected).T
                output_error = np.abs(error)
                
            else:
                #propogate error backwards through network (Multiply error by outgoing weights).
                #Then sum along columns(expect for bias column) in order to get the sum of error outgoing to each neuron in thep previous layer.
                error = self.weights[i].T * error
            
            #hadamard product of error and derivative of sigmoid function on output of each neuron. 
            error = np.multiply(error , self._derivative_tanh(self.weights[i-1]*outputs[i-1] + self.biases[i-1]))
        
        
            self.weights[i-1] -= self.learning_rate*error*outputs[i-1].T
            #only get array after the bais weight, the last weight(returns a subarrray not an element).
            self.biases[i-1] -= self.learning_rate*error
            
        return output_error
            
    def train(self, training_set, expected_set, error_margin =0.05 ,epochs=16000):
          
        for _ in range(epochs):
            error = 0.0
            for example, expected in zip(training_set, expected_set):
                error += self.backpropogation(example, expected)
            if error < error_margin:
                break
        
                
                

net = ANN(learning_rate=0.05)
'''
net.random_weights((2,2,2,1))
net.train([(0,1),(1,0),(1,1),(0,0)],[1,0,0,1])


print(net.forward((0,1)))
print(net.forward((1,1)))
print(net.forward((1,0)))
print(net.forward((0,0)))
'''
net.random_weights((2,2,2,1))
net.train([(0,1),(1,0),(1,1),(0,0)],[1,1,0,0])

#tanh performs better for making xor gate


print(net.forward((0,1)))
print(net.forward((1,0)))
print(net.forward((1,1)))
print(net.forward((0,0)))