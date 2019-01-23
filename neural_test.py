import numpy  as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# inputting the data
traning_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

#Entering output tranning data 
traning_outputs =  np.array([[0,1,1,0]])

np.random.seed(1)

#3 metrix 
synaptic_weights = 2 * np.random.random((3, 1)) -1

#printing the  inputs
print("Inputs are Entered")
print(traning_inputs)
#prining the synaptic weights
print("Random starting synaptic weights: ")
print(synaptic_weights)

#the main loop
for itration in range (1):
    input_layer = traning_inputs
    outputs =   sigmoid(np.dot(input_layer, synaptic_weights))

#printing the outputs
print("print outputs after the training: ")
print(outputs)
