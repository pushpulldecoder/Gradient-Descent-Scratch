# Gradient Descent from Scratch

<!-- > Looking down the misty path to uncertain destinations🌌🍀&nbsp;&nbsp;&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- x' <br><br><br>
 -->

It is a gradient descent algorithm for classification implemented from scratch using numpy library.


<img src = https://github.com/pushpulldecoder/Gradient-Descent-Scratch/blob/master/ann.jpg>

## Dependencies
- Numpy
- Matplotlib
- Pandas

## Importing Dataset
###### Datasets are first imported as pandas dataframe and then converted into numpy arrays
```python
train_data_frame = pd.read_csv('train_dataset.csv', header=None)
test_data_frame  = pd.read_csv('test_dataset.csv',  header=None)
```
```python
train_dataset = np.array(train_data_frame)
test_dataset  = np.array(test_data_frame)
```

## Splitting data into input and label
###### Here in MNIST, column 0 is label and all other are inputs
```python
train_lable = np.array([train_dataset[:, 0]])
train_data  = np.array(train_dataset[:, 1:785]).T

test_lable  = np.array([test_dataset[:, 0]]).T
test_data   = np.array(test_dataset[:, 1:785])
```
It is good practice to shuffle data at first<br>
<b>numpy.random.shuffle()</b> will shuffle array in place
```python
np.random.shuffle(train_dataset)
```

## Parameter Initialization
###### In This network, weights are initialized randomly while biases are initialized zero as a list of numpy array
```python
def __init__(self, size):
	self.biases  = [np.zeros([y, 1]) for y in size[1:]]
	self.weights = [np.random.randn(y, x)*0.01 for x, y in zip(size[:-1], size[1:])]
```

## Choosing Hyperparameter
<b><i>Mini Batch Size</i></b> is size of input data flowing through network at a time for calculating error as a whole<br>
Learning Rate <b><i>Alpha</i></b> decides the rate at which, weights and biases will update while back propagation<br>
Number of <b><i>Epochs</i></b> decides number of times, the whole dataset will be used to train the network<br>
<i>Set Mini Batch Size to 1/10<sup>th</sup> of total data available. And update it manually after every train of network to find its optimum value</i><br>
<i>Alpha should be selected such that learning isn't very slow as well as it didn't take long jump or else, network will start diverging from local minima</i><br>
<i>Number of epochs are selected such that network don't overfit itself over noise</i>

## Feed Forward
###### For layer l = 2, 3,..., L compute
- ###### z[l] = W[l].A[l-1] + B[l]
- ###### A[l]  =  σ(Z[l])
```python
def train_feed_forward(self, size, input, activators):
    self.z = [np.zeros([y, input]) for y in size[:]]
    i=0
    self.z[0] = input
    for bias, weight in zip(self.biases, self.weights):
        input = (np.dot(weight, input) + bias)
        self.z[i+1] = input
        input = activation(input)
        i=i+1
    return input
```

## Activation Functions
###### Applying activation functions will change nature of network from linear from to non linear so that it could fit the outputs more accurately or else, it would be no different than any linear regression
```python
def sigmoid(z, derivative=False):
    if derivative==True:
        return (activator.sigmoid(z=z, derivative=False) * (1 - activator.sigmoid(z=z, derivative=False)))
    return (1.0 / (1.0 + np.exp(-z)))
```
```python
def softmax(z, derivative=False):
    if derivative==True:
        return (activator.softmax(z=z, derivative=False) * (1 - activator.softmax(z=z, derivative=False)))
    return (np.exp(z) / np.sum(np.exp(z)))
```
```python
def tanh(z, derivative=False):
    if derivative==True:
        return (activator.tanh(z=z, derivative=False) * (1 - activator.tanh(z=z, derivative=False)))
    return (np.tanh(z))
```
```python
def relu(z, derivative=False):
    if derivative==True:
        der_z = np.zeros(z.shape)
        for i in range(len(z.shape)):
            for j in range(len(z[i])):
                if(z[i, j]>0):
                    der_z[i, j] = 1
        return der_z
    return (np.maximum(z, 0))
```

## Error and Loss Function
###### For error calculation, mean squared error is used
###### Output Error δ[L] = ∇aC ⊙ σ′(Z[L])
###### Mean Squared Error = (Predicted_value - Expected_value)<sup>2</sup>
```python
def loss(self, Y, Y_hat, derivative=False):
    if derivative==True:
        return (Y_hat-Y)
    return ((Y_hat - Y) ** 2)
```

## Backpropagation
In ANN, output will depend on every neuron it pass through<br>
For output layer, we have label according to which, it is possible to find it's expected value<br>
But for all other layers, there is no single solution available<br>
So, finding optimum value is little harder for that<br>
<img src = https://github.com/pushpull13/Gradient-Descent-Scratch/blob/master/backprop.jpg>
###### For each l=L−1,L−2,…,2 compute
###### δ<sup>l</sup>&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;&nbsp;&nbsp;(W<sup>l+1</sup>T . δ<sup>l+1</sup>) ⊙ σ′Z<sup>l</sup>
###### δ<sup>l</sup>&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;&nbsp;&nbsp;((W[l+1])T δ[l+1]) ⊙ σ′(Z[l])
```python
delta_nabla = self.find_nabla(size=size, activators=activators, mini_batch=mini_batch, mini_batch_size=mini_batch_size, y=y, alpha=alpha)
```
```python
y_hat = self.train_feed_forward(size=size, input=mini_batch, activators=activators, mini_batch_size=mini_batch_size)
delta_nabla_b = [np.zeros([y, 1]) for y in size[1:]]
delta_nabla_w = [np.zeros([y, x]) for x, y in zip(size[:-1], size[1:])]
```
```python
delta = self.loss(Y=y, Y_hat=y_hat, derivative=True) * activator.sigmoid(z=y_hat, derivative=True)
delta_nabla_b[-1] += np.sum(delta)
delta_nabla_w[-1] += np.dot(delta, self.z[-2].T)
```
```python
delta = np.dot(self.weights[layer_no].T, delta) * activator.sigmoid(z=self.z[layer_no-1], derivative=True)

delta_nabla_b[layer_no-1] += np.sum(delta)
delta_nabla_w[layer_no-1] += np.dot(delta, self.z[layer_no-2].T)

delta_nabla = [delta_nabla_b, delta_nabla_w]
```
###### Updating Weight and Biases
###### ∂C/∂W<sup>l</sup><sub>j, k</sub> &nbsp;&nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;&nbsp;&nbsp; A<sup>l-1</sup><sub>k</sub> . δ<sup>l</sup><sub>j</sub>
###### ∂C/∂B<sup>l</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=&nbsp;&nbsp;&nbsp;&nbsp;δ<sup>l</sup><sub>j</sub>
```python
self.biases  = [b-((alpha/mini_batch_size)*n_b) for b, n_b in zip(self.biases, delta_nabla[0])]
self.weights = [w-((alpha/mini_batch_size)*n_w) for w, n_w in zip(self.weights, delta_nabla[1])]

```

## Creating Network
In <b><i>size_layers</i></b>, define number of neurons on every layer of network and activation functions of every layer in <b><i>activations</i></b>
###### Network will work if more layers are added or deleted
```python
neuron_layer = {"size_layers": [784, 2800, 10], "activations": ["tanh", "sigmoid"] }
my_network = network(neuron_layer["size_layers"])
```

## Training Network
```python
my_network.grad_descn(size=neuron_layer["size_layers"], expected_value=train_lable, training_data=train_data, activators=neuron_layer["activations"], alpha=0.01, mini_batch_size=2000, epochs=40)
```

## Testing Network
```python
result = test_feed_forward(size=neuron_layer["size_layers"], input=test_data.T, activators=neuron_layer["activations"])

no_trues = 0

for i in range(len(test_data)):
    max_ans = result[0, i]
    max_ind = 0
    for j in range(10):
        if(result[j, i]>max_ans):
            max_ind = j
            max_ans = result[j, i]
    if(test_lable[i]==max_ind):
        no_trues+=1
```
