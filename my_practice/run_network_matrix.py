import network_matrix
import mnist_loader

if if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network_matrix.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)