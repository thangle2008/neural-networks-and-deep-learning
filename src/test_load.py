import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU

training_data, validation_data, test_data = network3.load_data_shared()
#expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
mini_batch_size = 10

net = Network.load("out.txt")
print net.accuracy(test_data, mini_batch_size)
