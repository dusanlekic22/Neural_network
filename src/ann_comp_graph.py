from abc import abstractmethod
import math
from os import access
import random
import copy
from numpy.lib.function_base import append
import pandas as pd
from matplotlib import pyplot
import random
from time import sleep
from tqdm import tqdm
from tqdm import trange 
from numpy.random import RandomState
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN
import os

random.seed(1337)


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dz):
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] je ulaz, x[1] je tezina

    def forward(self, x):
        self.x = x
        # TODO 1: implementirati forward-pass za mnozac  
        return self.x[0]*self.x[1]

    def backward(self, dz):
        # TODO 1: implementirati backward-pass za mnozac
        return [dz*self.x[1],dz*self.x[0]]


# MultiplyNode tests
mn_test = MultiplyNode()
assert mn_test.forward([2., 3.]) == 6., 'Failed MultiplyNode, forward()'
assert mn_test.backward(-2.) == [-2.*3., -2.*2.], 'Failed MultiplyNode, backward()'
print('MultiplyNode: tests passed')


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x je vektor, odnosno niz skalara

    def forward(self, x):
        self.x = x
        # TODO 2: implementirati forward-pass za sabirac
        return sum(self.x)

    def backward(self, dz):
        # TODO 2: implementirati backward-pass za sabirac
        return [dz for xx in self.x]


# SumNode tests
sn_test = SumNode()
assert sn_test.forward([1., 2., -2, 5.]) == 6., 'Failed SumNode, forward()'
assert sn_test.backward(-2.) == [-2.]*4, 'Failed SumNode, backward()'
print('SumNode: tests passed')


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x je skalar

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        # TODO 3: implementirati backward-pass za sigmoidalni cvor 
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        # TODO 3: implementirati sigmoidalnu funkcij
        try:
            return 1. / (1. + math.exp(-x))
        except:
            return float('inf')


# SigmoidNode tests
sign_test = SigmoidNode()
assert sign_test.forward(1.) == 0.7310585786300049, 'Failed SigmoidNode, forward()'
assert sign_test.backward(-2.) == -2.*0.7310585786300049*(1.-0.7310585786300049), 'Failed SigmoidNode, backward()'
print('SigmoidNode: tests passed')

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

class LinNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._lin(self.x)

    def backward(self, dz):
        return dz 

    def _lin(self, x):
        return x        

class TanhNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an i

    def forward(self, x):
        self.x = x
        return self._tanh(self.x)

    def backward(self, dz):
        return dz* (1-self._tanh(self.x)**2)

    def _tanh(self, x):
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        except:
            return float('inf')      


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs  # moramo da znamo kolika ima ulaza da bismo znali koliko nam treba mnozaca
        self.multiply_nodes = []  # lista mnozaca
        self.sum_node = SumNode()  # sabirac

        # TODO 4: napraviti n_inputs mnozaca u listi mnozaca, odnosno mnozac za svaki ulaz i njemu odgovarajucu tezinu
        # za svaki mnozac inicijalizovati tezinu na broj iz normalne (gauss) raspodele sa st. devijacijom 0.1
        for i in range(self.n_inputs):
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]
            self.multiply_nodes.append(mn)

        # TODO 5: dodati jos jedan mnozac u listi mnozaca, za bias
        # bias ulaz je uvek fiksiran na 1.
        # bias tezinu inicijalizovati na broj iz normalne (gauss) raspodele sa st. devijacijom 0.01
        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        # TODO 6: ako ulazni parametar funckije 'activation' ima vrednosti 'sigmoid',
        # inicijalizovati da aktivaciona funckija bude sigmoidalni cvor
        if activation=='sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        elif activation == 'lin':
            self.activation_node = LinNode()
        elif activation == 'tanh':
            self.activation_node = TanhNode()
        else:
            raise RuntimeError('Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x je vektor ulaza u neuron, odnosno lista skalara
        x = copy.copy(x)
        x.append(1.)  # uvek implicitino dodajemo bias=1. kao ulaz

        # TODO 7: implementirati forward-pass za vestacki neuron
        # u x se nalaze ulazi i bias neurona
        # iskoristi forward-pass za mnozace, sabirac i aktivacionu funkciju da bi se dobio konacni izlaz iz neurona
        for_sum = []
        for i,xx in enumerate(x):
            inp = [x[i],self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))
        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        #print(dz)
        d = dz[0] if type(dz[0]) == float else sum(dz)  # u d se nalazi spoljasnji gradijent izlaza neurona

        # TODO 8: implementirati backward-pass za vestacki neuron
        # iskoristiti backward-pass za aktivacionu funkciju, sabirac i mnozace da bi se dobili gradijenti tezina neurona
        # izracunate gradijente tezina ubaciti u listu dw
        act = self.activation_node.backward(d)
        summed_act = self.sum_node.backward(act)
        for i, bb in enumerate(summed_act):
            dw.append(self.multiply_nodes[i].backward(bb)[1])

        self.gradients.append(dw)
        return dw

    def update_weights(self, learning_rate, momentum):
        # azuriranje tezina vestackog neurona
        # learning_rate je korak gradijenta

        # TODO 11: azurirati tezine neurona (odnosno azurirati drugi parametar svih mnozaca u neuronu)
        # gradijenti tezina se nalaze u list self.gradients
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_grad = sum([grad[i] for grad in self.gradients]) / len(self.gradients)
            delta = learning_rate*mean_grad + momentum * self.previous_deltas[i]
            self.multiply_nodes[i].x[1] -= delta
            self.previous_deltas[i] = delta

        self.gradients = []  # ciscenje liste gradijenata (da sve bude cisto za sledecu iteraciju)


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs  # broj ulaza u ovaj sloj neurona
        self.n_neurons = n_neurons  # broj neurona u sloju (toliko ce biti i izlaza iz ovog sloja)
        self.activation = activation  # aktivaciona funkcija neurona u ovom sloju

        self.neurons = []
        # konstruisanje sloja nuerona
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x je vektor, odnosno lista "n_inputs" elemenata
        layer_output = []
        # forward-pass za sloj neurona je zapravo forward-pass za svaki neuron u sloju nad zadatim ulazom x
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz je vektor, odnosno lista "n_neurons" elemenata
        dd = []
        # backward-pass za sloj neurona je zapravo backward-pass za svaki neuron u sloju nad
        # zadatim spoljasnjim gradijentima dz
        for i, neuron in enumerate(self.neurons):
            neuron_dz = [d[i] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            dd.append(neuron_dz[:-1])  # izuzimamo gradijent za bias jer se on ne propagira unazad

        return dd

    def update_weights(self, learning_rate, momentum):
        # azuriranje tezina slojeva neurona je azuriranje tezina svakog neurona u tom sloju
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        self.layers = []  # neuronska mreza se sastoji od slojeva neurona

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x je vektor koji predstavlja ulaz u neuronsku mrezu
        # TODO 9: implementirati forward-pass za celu neuronsku mrezu
        # ulaz za prvi sloj neurona je x
        # ulaz za sve ostale slojeve izlaz iz prethodnog sloja
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)
        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        # TODO 10: implementirati forward-pass za celu neuronsku mrezu
        # spoljasnji gradijent za izlazni sloj neurona je dz
        # spoljasnji gradijenti za ostale slojeve su izracunati gradijenti iz sledeceg sloja
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)
        return next_layer_dz
        

    def update_weights(self, learning_rate, momentum):
        # azuriranje tezina neuronske mreze je azuriranje tezina slojeva
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate=0.1, momentum=0.0, nb_epochs=10, shuffle=False, verbose=0):
        assert len(X) == len(Y)

        hist = []  # za plotovanje funkcije greske kroz epohe
        for epoch in trange(nb_epochs):
            if shuffle:  # izmesati podatke
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            for x, y in zip(X, Y):
                y_pred = self.forward(x)  # forward-pass da izracunamo izlaz
                y_target = y  # zeljeni izlaz
                grad = 0.0
                for p, t in zip(y_pred, y_target):
                    total_loss += 0.5 * (t-p) ** 2.  # funkcija greske je kvadratna greska
                    #total_loss +=self.CrossEntropy(p, t)
                    grad+=-(t-p)  # gradijent funkcije greske u odnosu na izlaz
                    #print(grad)
                # backward-pass da izracunamo gradijente tezina
                self.backward([[grad]])
                # azuriranje tezina na osnovu izracunatih gradijenata i koraka "learning_rate"
                self.update_weights(learning_rate, momentum)
                
                

            if verbose == 1:
                print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
            hist.append(total_loss)
        
        
        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)

    def CrossEntropy(self,yHat, y):
        if y == 1:
          return -math.log(yHat)
        else:
          return -math.log(1 - yHat)

def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MaxAbsScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled)

def concat (df,columns):
    ulazi = df
    for c in columns:
        ulazi = pd.concat([ulazi, pd.get_dummies(ulazi[c], prefix=c, dummy_na=False)], axis=1).drop([c], axis=1)     
    return ulazi
    
def extractDigits(lst):
    return [[el] for el in lst]

def analisys_plot(x,y,x_axis='x_axis',y_axis='y_axis',title=None):
    x_pos = [i for i, _ in enumerate(x)]
    pyplot.style.use('ggplot')
    pyplot.bar(x_pos, y, color='green')
    pyplot.xlabel(x_axis)
    pyplot.ylabel(y_axis)
    pyplot.title(title)
    pyplot.xticks(x_pos, x)
    pyplot.show()

def analysis(path,col,types):
    lst = [0] * len(types)
    fst = [0] * len(types)
    df = pd.read_csv(path)
    for row in df.itertuples(index=True, name='Pandas'):
        if  row.stroke==1:
            for t in types:
                if row[df.columns.get_loc(col)+1]==t:
                    lst[types.index(t)]+=1
        else:
            for t in types:
                if row[df.columns.get_loc(col)+1]==t:
                    fst[types.index(t)]+=1
    return [i / j for i, j in zip(lst, fst)]

if __name__ == '__main__':
    
    path = os.path.realpath(r'Neural_network\data\kolokvijum.csv')
    #pod a)
    #analysis(path)
    types_smokers = ['Smoker','Non Smoker','Former Smoker']
    smokers_strokes=analysis(path,"smoking_status",['smokes','never smoked','formerly smoked'])
    analisys_plot(types_smokers,smokers_strokes,"Types of smokers","Strokes","Strokes in correlation to smoking history")

    genders = ['Male','Female','Other']
    genders_strokes = analysis(path,"gender",['Male','Female','Other'])
    analisys_plot(genders,genders_strokes,"Genders","Strokes","Strokes in correlation to genders")

    hypertension = ['Had hypertension',"Didn't have hypertension"]
    hypertension_strokes = analysis(path,"hypertension",[1,0])
    analisys_plot(hypertension,hypertension_strokes,"Hypertension","Strokes","Strokes in correlation to hypertension history")

    heart_disease = ['Had heart disease',"Didn't have heart disease"]
    heart_disease_strokes = analysis(path,"heart_disease",[1,0])
    analisys_plot(heart_disease,heart_disease_strokes,"Heart disease","Strokes","Strokes in correlation to heart disease history")

    residence_type = ['Urban','Rural']
    residence_type_strokes = analysis(path,"Residence_type",['Urban','Rural'])
    analisys_plot(residence_type,residence_type_strokes,"Residence_type","Strokes","Strokes in correlation to residence type") 
    #pod b)
    nn = NeuralNetwork()                        
    
    no_id = ["gender", "age","heart_disease","hypertension","ever_married","work_type","Residence_type","bmi","avg_glucose_level","smoking_status","stroke"]
    df = pd.read_csv(path,usecols=no_id)
    df.fillna(0,inplace=True)
    ada = ADASYN()
    list = []
    for i in range(22):
       list.append(i)
    list.remove(5)
    
    df=concat(df,['gender','ever_married','Residence_type','work_type','smoking_status'])
    df.to_csv(path_or_buf=os.path.realpath(r'Neural_network\data\in0.csv'),index=False)
    #Oversampling
    X_resampled, y_resampled = ada.fit_resample(df.iloc[:,list], df['stroke'])
    df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    df.to_csv(path_or_buf=os.path.realpath(r'Neural_network\data\in0.csv'),index=False)
    #Deljenje podataka na test i training sa osiguranjem da se nalazi isti odnos 1 i 0 u test i training
    stroke1 = df[df["stroke"] == 1]
    stroke0 = df[df["stroke"] == 0]
    inputs_len = len(df.columns)-1

    stroke1_ulazi = stroke1.drop('stroke', axis=1)
    stroke1_izlaz = stroke1["stroke"]
    stroke0_ulazi = stroke0.drop('stroke', axis=1)
    stroke0_izlaz = stroke0["stroke"]
    stroke1_ulazi=normalize(stroke1_ulazi)
    stroke0_ulazi=normalize(stroke0_ulazi)
    ktrain_ulazi1 = stroke1_ulazi.sample(frac=0.7, random_state=RandomState())
    ktrain_ulazi0 =  stroke0_ulazi.sample(frac=0.7, random_state=RandomState())
    ktest_ulazi1=stroke1_ulazi.loc[~stroke1_ulazi.index.isin(ktrain_ulazi1.index)]
    ktest_ulazi0= stroke0_ulazi.loc[~ stroke0_ulazi.index.isin(ktrain_ulazi0.index)]
    ktrain_izlazi1 = stroke1_izlaz.sample(frac=0.7, random_state=RandomState())
    ktrain_izlazi0 =  stroke0_izlaz.sample(frac=0.7, random_state=RandomState())
    ktest_izlazi1=stroke1_izlaz.loc[~stroke1_izlaz.index.isin(ktrain_izlazi1.index)]
    ktest_izlazi0=stroke0_izlaz.loc[~stroke0_izlaz.index.isin(ktrain_izlazi0.index)]
    ktrain_ulazi = pd.concat([ktrain_ulazi1,ktrain_ulazi0])
    ktrain_izlaz = pd.concat([ktrain_izlazi1,ktrain_izlazi0])
    ktrain_ulazi.to_csv(path_or_buf=os.path.realpath(r'Neural_network\data\in.csv'),index=False)
    ktrain_izlaz.to_csv(path_or_buf=os.path.realpath(r'Neural_network\data\out.csv'),index=False)
    ktest_ulazi= pd.concat([ktest_ulazi1,ktest_ulazi0])
    ktest_izlaz = pd.concat([ktest_izlazi1,ktest_izlazi0])
    ktest_ulazi.to_csv(path_or_buf=os.path.realpath(r'Neural_network\data\test_in.csv'),index=False)
    ktest_izlaz.to_csv(path_or_buf=os.path.realpath(r'Neural_network\data\test_out.csv'),index=False)
    
    nn.add(NeuralLayer(inputs_len, inputs_len, 'sigmoid'))
    nn.add(NeuralLayer(inputs_len, 10, 'tanh'))
    nn.add(NeuralLayer(10, 1, 'sigmoid'))

    history = nn.fit(ktrain_ulazi.values.tolist(), extractDigits(ktrain_izlaz.values.tolist()), learning_rate=0.01, momentum=0.99, nb_epochs=30, shuffle=True, verbose=0)
    pyplot.plot(history)
    pyplot.show()
    tp = tn= fp =fn = 0
    accuracy_plot = []
    test_X = ktest_ulazi.values.tolist()
    test_Y = ktest_izlaz.values.tolist()
    counter = 0
    for i,j in zip(test_X,test_Y):
        if(nn.predict(i)[0]>0.5):
            print([j,nn.predict(i)[0]])
            counter+=1
        if j==1 :
            if nn.predict(i)[0] > 0.5 :
                tp+=1
            elif nn.predict(i)[0] < 0.5:
                fn+=1
        elif j==0:
            if nn.predict(i)[0] < 0.5:
                tn+=1
            elif nn.predict(i)[0] > 0.5:
                fp+=1
        accuracy_plot.append((tp+tn)*100/(tp+fp+fn+tn))     
    print(counter)
    try:
        accuracy = (tp+tn)/(tp+fp+fn+tn)
    except:
        accuracy = 0
    try:
        precision = tp / (tp+fp)
    except:
        precision = 0
    try:
        recall = tp / (tp+fn)
    except:
        recall=0
    try:
        f1 = (precision*recall)/(precision+recall)
    except:
        f1 = 0
    pyplot.plot(accuracy_plot)
    pyplot.show()
    print('Accuracy',round(accuracy*100,2),'%')
    print('Precision:',round(precision,2))
    print('Recall:',round(recall,2))
    print('F1 je',round(f1,2))