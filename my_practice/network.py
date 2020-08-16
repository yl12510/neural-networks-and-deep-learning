import numpy as np

class Network(object): 

    def __init__(self, sizes): 
        # sizes will be a list of numbers. e.g. net = Network([2,3,1])
        # The length of the list will be the number of layers of the net
        # each number represents the number of neurons of each layer  
        self.num_layers = len(sizes)
        self.sizes = sizes 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # no biases for the input layer 
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a): 
        # for an input, compute the output of network based on the current weights and bias
        for w, b in zip(self.weights, self.biases): 
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None): 
        # train the neural network by using mini-batch stochastic gradient descent 
        # training_data: tuples like (x,y), where x represents each training input, 
        #                and y represents the desired output 
        # 
        # if test data is provided, the network will be evaluated after each epoch of training 
        
        training_data = list(training_data)
        n = len(training_data)

        if test_data: 
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs): 
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: 
                self.update_mini_batch(mini_batch, eta)
            if test_data: 
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else: 
                print(f'Epoch {j} finished')

    def update_mini_batch(self, mini_batch, eta): 
        # mini_batch, a list of tuples (x,y) 
        # eta: learning rate
         
        # initialise nabla weights and biases as zeros 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
         
        for x, y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y): 
        # return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward 
        activation = x 
        activations = [x] # list to store all layers of activations 
        zs = [] # list to store all layers of z vectors 

        for b, w in zip(self.biases, self.weights): 
            z = np.dot(w, activation) + b # calculate the current layer's input z
            zs.append(z)

            activation = sigmoid(z) # update activation for the next layer
            activations.append(activation)

        # backward pass 
        # last layer first 
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # from the second last layer
        for l in range(2, self.num_layers): 
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y): 
        return (output_activations - y)

    def evaluate(self, test_data): 
        # return the number of correct outputs 
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z): 
    # define sigmoid function
    return 1.0/(1.0+np.exp(-z)) 

def sigmoid_prime(z): 
    return sigmoid(z) * (1 - sigmoid(z))