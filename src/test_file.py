import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 1.0, lmbda = 1000.0, evaluation_data= \
	validation_data, \
	monitor_evaluation_accuracy=True, \
	adjust_learning_rate=True)
