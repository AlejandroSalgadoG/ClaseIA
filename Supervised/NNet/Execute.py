from NNet import NNet

arch = [2, 2, 2]
input_data = [0.05, 0.1]
labels = [0.01, 0.99]
eta = 0.5

weights = [ [], [] ]
weights[0] = [ [0.15, 0.25],
               [0.20, 0.30],
               [0.35, 0.35] ] # Esta ultima linea son los pesos de los bias

weights[1] = [ [0.40, 0.50],
               [0.45, 0.55],
               [0.60, 0.60] ] # Esta ultima linea son los pesos de los bias

nnet = NNet(arch)

weights = nnet.train(input_data, labels, weights, eta)
print(weights)

output = nnet.predict(input_data, weights)
print(output)
