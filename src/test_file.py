import network3
import matplotlib.pyplot as plt
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU

training_data, validation_data, test_data = \
            network3.load_data_shared('../data/bird_image_full.pkl.gz')

mini_batch_size = 10

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2,2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2,2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU, p_dropout=0.5),        
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

costs, train_acc = net.SGD(training_data, 200, mini_batch_size, 0.1, validation_data, test_data, lmbda=0.4,
        diminishing_lr=True) 
it = range(0, 1000 * len(costs), 1000)
plt.plot(it, costs)
plt.ylabel('cost')
plt.show()

#net.save("out.txt")
