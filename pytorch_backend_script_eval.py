import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
# Flask app for real-time data transmission
app = Flask(__name__)

# Define the Spiking Neural Network model
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, num_neurons, input_size, output_size, hidden_size):
        super(SpikingNeuralNetwork, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(self.device)

        # Initialize other parameters
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.weights = torch.rand(num_neurons, num_neurons, device=self.device) * 2 - 1  # [-1, 1]
        self.last_spike_time = torch.zeros(num_neurons, device=self.device)
        self.current_time = 0
        self.membrane_potential = torch.zeros(num_neurons, device=self.device)
        self.membrane_decay = 0.5

        # Hodgkin-Huxley parameters
        self.g_Na = 120.0  # mS/cm^2
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0  # mV
        self.E_K = -77.0
        self.E_L = -54.4
        self.n = torch.zeros(num_neurons, device=self.device)
        self.m = torch.zeros(num_neurons, device=self.device)
        self.h = torch.zeros(num_neurons, device=self.device)

    def forward(self, x):
        x = self.update_hodgkin_huxley_dynamics(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def update_hodgkin_huxley_dynamics(self, input_signal):
        # Convert input_signal to a tensor if it's not already
        input_signal = input_signal.to(self.device, dtype=self.fc1.weight.dtype)

        # Hodgkin-Huxley model dynamics
        dt = 0.01
        self.membrane_potential = self.membrane_decay * self.membrane_potential + torch.sigmoid(self.fc1(input_signal))
        spiking_neurons = self.membrane_potential > 1.0
        self.membrane_potential[spiking_neurons] = 0

        # Update gating variables
        alpha_n = 0.01 * (self.membrane_potential + 55) / (1 - torch.exp(-(self.membrane_potential + 55) / 10))
        beta_n = 0.125 * torch.exp(-(self.membrane_potential + 65) / 80)
        alpha_m = 0.1 * (self.membrane_potential + 40) / (1 - torch.exp(-(self.membrane_potential + 40) / 10))
        beta_m = 4.0 * torch.exp(-(self.membrane_potential + 65) / 18)
        alpha_h = 0.07 * torch.exp(-(self.membrane_potential + 65) / 20)
        beta_h = 1 / (1 + torch.exp(-(self.membrane_potential + 35) / 10))

        self.n = self.n.view(1, -1)
        self.n += dt * (alpha_n * (1 - self.n) - beta_n * self.n)
        self.m = self.m.view(1, -1)
        self.m += dt * (alpha_m * (1 - self.m) - beta_m * self.m)
        self.h = self.h.view(1, -1)
        self.h += dt * (alpha_h * (1 - self.h) - beta_h * self.h)

        self.I_Na = self.g_Na * self.m**3 * self.h * (self.membrane_potential - self.E_Na)
        self.I_K = self.g_K * self.n**4 * (self.membrane_potential - self.E_K)
        self.I_L = self.g_L * (self.membrane_potential - self.E_L)

        # STDP-based synaptic weight update
        self.update_weights(spiking_neurons)

        return self.membrane_potential

    def update_weights(self, spiking_neurons):
        # Spike-Timing Dependent Plasticity (STDP) implementation
        tau_plus = 15.0
        tau_minus = 25.0
        A_plus = 0.02
        A_minus = 0.015

        time_since_last_spike = self.current_time - self.last_spike_time
        for i in range(spiking_neurons.size(0)):
            for j in range(spiking_neurons.size(0)):
                if spiking_neurons[i].any() or spiking_neurons[j].any():
                    delta_t = time_since_last_spike[i] - time_since_last_spike[j]
                    if delta_t > 0:
                        delta_w = -A_minus * torch.exp(-torch.abs(delta_t) / tau_minus)
                    else:
                        delta_w = A_plus * torch.exp(-torch.abs(delta_t) / tau_plus)
                    self.weights[i, j] += delta_w

        self.weights = torch.clamp(self.weights, -1, 1)
        spiking_neurons = spiking_neurons.view(self.last_spike_time.shape)
        self.last_spike_time[spiking_neurons] = self.current_time        
        self.current_time += 1
        
        
        
# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize the SNN
num_neurons = 1000
input_size = 1000  # Size of MNIST images
output_size = 1000  # Number of classes in MNIST
hidden_size = 500  # Size of hidden layer
snn = SpikingNeuralNetwork(num_neurons, input_size, output_size, hidden_size).to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(snn.parameters(), lr=0.01)


# Initialize the SNN and load the model weights
snn = SpikingNeuralNetwork(num_neurons, input_size, output_size, hidden_size)
snn.load_state_dict(torch.load('snn_modelv4.pth', map_location=device))
snn.eval()  # Set to evaluation mode, if you're not training

            
            
@app.route('/get_model_state', methods=['GET'])
def get_model_state():
    # Simulate a training step with a single batch from the dataset
    images, labels = next(iter(train_loader))
    outputs, loss = train_step(snn, images, labels, optimizer, criterion)

    # Extract the weights, biases, and neuron states
    weights = snn.fc1.weight.data.cpu().numpy().tolist()
    biases = snn.fc1.bias.data.cpu().numpy().tolist()
    neuron_states = outputs.data.cpu().numpy().tolist()

    # Prepare data for transmission
    data = {
        'inputs': images.view(-1, 28*28).data.cpu().numpy().tolist(),
        'neuron_states': neuron_states,
        'weights': weights,
        'biases': biases,
        'loss': loss.item()
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    # Load some sample data for evaluation, or use your own custom data
    sample_data = torch.randn(1, input_size).to(device)  # Example input tensor

    # Forward pass to get neuron states without updating weights
    with torch.no_grad():  # Important: this disables gradient computation
        neuron_states = snn(sample_data)

    # Extract the neuron states
    neuron_states = neuron_states.data.cpu().numpy().tolist()

    # Prepare data for transmission
    data = {
        'neuron_states': neuron_states
    }

    return jsonify(data)