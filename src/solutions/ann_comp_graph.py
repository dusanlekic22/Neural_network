from __future__ import print_function

from abc import abstractmethod
import math
import random
import copy
import pandas as pd

from numpy.lib.function_base import append
import pandas as pd

from numpy.random import RandomState
from matplotlib import pyplot
import random
from tqdm import trange 

random.seed(1337)


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):  # x is an array of scalars
        pass

    @abstractmethod
    def backward(self, dz):  # dz is a scalar
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] is input, x[1] is weight

    def forward(self, x):
        self.x = x
        return self.x[0] * self.x[1]

    def backward(self, dz):
        return [dz * self.x[1], dz * self.x[0]]


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x is in an array of inputs

    def forward(self, x):
        self.x = x
        return sum(self.x)

    def backward(self, dz):
        return [dz for xx in self.x]


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        try:
            return 1. / (1. + math.exp(-x))
        except:
            return float('inf')


class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0.)

    def _relu(self, x):
        return max(0., x)


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs
        self.multiply_nodes = []  # for inputs and weights
        self.sum_node = SumNode()  # for sum of inputs*weights

        for n in range(n_inputs):  # collect inputs and corresponding weights
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]  # init input weights
            self.multiply_nodes.append(mn)

        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        else:
            raise RuntimeError('Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x is a vector of inputs
        x = copy.copy(x)
        x.append(1.)  # for bias
        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))

        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        b = dz[0] if type(dz[0]) == float else sum(dz)
        b = self.activation_node.backward(b)
        b = self.sum_node.backward(b)
        for i, bb in enumerate(b):
            dw.append(self.multiply_nodes[i].backward(bb)[1])

        self.gradients.append(dw)
        return dw

    def update_weights(self, learning_rate, momentum):
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_gradient = sum([grad[i] for grad in self.gradients]) / len(self.gradients)
            delta = learning_rate*mean_gradient + momentum*self.previous_deltas[i]
            self.previous_deltas[i] = delta
            self.multiply_nodes[i].x[1] -= delta

        self.gradients = []


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        self.neurons = []
        # construct layer
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x is a vector of "n_inputs" elements
        layer_output = []
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz is a vector of "n_neurons" elements
        b = []
        for idx, neuron in enumerate(self.neurons):
            neuron_dz = [d[idx] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            b.append(neuron_dz[:-1])

        return b  # b is a vector of "n_neurons" elements

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        # construct neural network
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x is a vector which is an input for neural net
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)

        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)

        return next_layer_dz

    def update_weights(self, learning_rate, momentum):
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate, momentum, nb_epochs, shuffle=False, verbose=0):
        assert len(X) == len(Y)

        hist = []
        for epoch in range(nb_epochs):
            if shuffle:
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            for x, y in zip(X, Y):
                # forward pass to compute output
                pred = self.forward(x)
                # compute loss
                grad = 0.0
                for o, t in zip(pred, y):
                    total_loss += (t - o) ** 2.
                    grad += -(t - o)
                # backward pass to compute gradients
                self.backward([[grad]])
                # update weights with computed gradients
                self.update_weights(learning_rate, momentum)

            hist.append(total_loss)
        if verbose == 1:
                print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)

    def load_from_file(self, file_path):
        """
        Ucitavanje table iz fajla.
        :param file_path: putanja fajla.
        """
        board_f = open(file_path, 'r')
        row = board_f.readline().strip('\n')
        self.data = []
        while row != '':
            self.data.append(list(row.split(',')))
            row = board_f.readline().strip('\n')
        board_f.close()


if __name__ == '__main__':
    nn = NeuralNetwork()
    # TODO 12: konstruisati neuronsku mrezu za resavanje XOR problema
    nn.add(NeuralLayer(2, 2, 'sigmoid'))
    nn.add(NeuralLayer(2, 1, 'sigmoid'))    
    mm = NeuralNetwork()

    ulaz = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\occupancy_train.csv',usecols=[2,3,4,5,6])
    izlaz = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\occupancy_train.csv',usecols=[7])
    test_ulaz = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\occupancy_test.csv',usecols=[2,3,4,5,6])
    test_izlaz = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\occupancy_test.csv',usecols=[7])
    #print(test_ulaz.values.tolist())
    longitude = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\skincancer.csv',usecols=[4])
    latitude = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\skincancer.csv',usecols=[1])
    mortitude = pd.read_csv (r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\skincancer.csv',usecols=[2])
    # obucavajuci skup
    #X = [[0., 0.],
    #     [1., 0.],
    #     [0., 1.],
    #     [1., 1.]]
    #Y = [[0., 0.],
    #     [1., 0.],
    #     [1., 1.],
    #     [0., 1.]]
    #Priprema za k2
    k_kuca = pd.read_csv(r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\dataset.csv',usecols=["house"])
    kuca = []
    for i in k_kuca.values:
       if i[0] not in kuca:
           kuca.append(i[0])
    dict= {}
    z=0
    for i in kuca:
        dict[i] = z
        z+=1
    #print (dict)

    col_list = ["male", "popularity","book1","book2","book3","book4","book5","isNoble","numDeadRelations","house"]
    k_ulazi = pd.read_csv(r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\dataset.csv',usecols=col_list)
    k_izlaz = pd.read_csv(r'C:\Users\Admin\Desktop\ori-2021-e2-public-main\06-ann-comp-graph\data\dataset.csv',usecols=["isAlive"])
    k_ulazi.house = [dict[item] for item in k_ulazi.house]
    ktrain_ulazi=k_ulazi.sample(frac=0.8, random_state=RandomState())
    ktrain_izlaz=k_izlaz.sample(frac=0.8, random_state=RandomState())
    ktest_ulazi=k_ulazi.loc[~k_ulazi.index.isin(ktrain_ulazi.index)]
    ktest_izlaz=k_izlaz.loc[~k_izlaz.index.isin(ktrain_izlaz.index)]

    print('Izaberi mrezu:')
    print('1.Prisutnost osobe u sobi u odnosu na parametre sobe')
    print('2.Stopa smrtnosti od raka kože u odnosu na geografsku sirinu/duzinu')
    print('3.Game of thrones')
    X=None
    Y=None
    x=input()
    if x=='1':
        X = ulaz.values.tolist()
        Y = izlaz.values.tolist()   
        X = [[0., 0.],
             [1., 0.],
             [0., 1.],
             [1., 1.]]
        Y = [[0., 0.],
             [1., 0.],
             [1., 1.],
             [0., 1.]]
        mm.add(NeuralLayer(2,2,'sigmoid'))
        mm.add(NeuralLayer(2,1,'sigmoid'))  
    elif x=='2':
        print('Po 1.sirini/2.duzini?:')
        if input()=='1':
            X = longitude.values.tolist()
        elif input()=='2':
            X = latitude.values.tolist()
        Y = mortitude.values.tolist()  
        mm.add(NeuralLayer(1,10,'sigmoid'))
        mm.add(NeuralLayer(10,1,'sigmoid'))
    elif x=='3': 
        X = ktrain_ulazi.values.tolist()
        Y = ktrain_izlaz.values.tolist()
        #print(X)
        mm.add(NeuralLayer(10,10,'sigmoid'))
        mm.add(NeuralLayer(10,1,'sigmoid'))
    else:
        print("Niste izabrali mrezu")
    
    if input("Treniraj mrezu?(y/n)") == 'y':
        history = mm.fit(X, Y, learning_rate=0.1, momentum=0.9, nb_epochs=3000, shuffle=True, verbose=0)
    #X = longitude.values.tolist()
    #Y = mortitude.values.tolist()
    # obucavanje neuronske mreze


    # provera da li radi
    print(nn.predict([0., 0.]))
    print(nn.predict([1., 0.]))
    print(nn.predict([0., 1.]))
    print(nn.predict([1., 1.]))
    print('Izaberi testiranje:')
    print('1.Prisutnost osobe u sobi u odnosu na parametre sobe')
    print('2.Game of thrones')
    tacno = 0
    if input() == '1':
        test_X = test_ulaz.values.tolist()
        test_Y = test_izlaz.values.tolist()
        for i,j in zip(test_X,test_Y):
            #print(j[0]-mm.predict(i)[0])
            if -0.5 < j[0]-mm.predict(i)[0] < 0.5:
                tacno+=1
        odnos = tacno / len(test_X)
        tacnost = odnos * 100
        print('Tacnost je',round(tacnost,2),'%')
    elif input() == '2':
        test_X = ktest_ulazi.values.tolist()
        test_Y = ktest_izlaz.values.tolist()
        for i,j in zip(test_X,test_Y):
            #print(j[0]-mm.predict(i)[0])
            if -0.5 < j[0]-mm.predict(i)[0] < 0.5:
                tacno+=1
        odnos = tacno / len(test_X)
        tacnost = odnos * 100
        print('Tacnost je',round(tacnost,2),'%')
    # plotovanje funkcije greske
    pyplot.plot(history)
    pyplot.show()
   
   
