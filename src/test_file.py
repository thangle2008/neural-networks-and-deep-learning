import network3
import matplotlib.pyplot as plt
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU

training_data, validation_data, test_data = \
            network3.load_data_shared('../data/bird_image.pkl.gz')
dim = 128

#expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
mini_batch_size = 10

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, dim, dim),
                      filter_shape=(20, 1, 9, 9),
                      poolsize=(2,2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 60, 60),
                      filter_shape=(40, 20, 9, 9),
                      poolsize=(2,2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*26*26, n_out=100, activation_fn=ReLU),        
        SoftmaxLayer(n_in=100, n_out=9)], mini_batch_size)

costs = net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1) 
#mb = range(0, 1000 * len(costs), 1000)
#plt.plot(mb, costs)
#plt.ylabel('cost')
#plt.show()

#net.save("out.txt")
