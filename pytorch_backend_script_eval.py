import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, render_template, request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import logging
from flask_cors import CORS

import logging
logging.basicConfig(level=logging.INFO)





# Flask app for real-time data transmission
app = Flask(__name__)
CORS(app)# Define the Spiking Neural Network model
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, num_neurons, input_size, output_size, hidden_size):
        super(SpikingNeuralNetwork, self).__init__()

        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Define layers and move them to the device
        self.fc1 = nn.Linear(input_size, hidden_size).to('cuda:0')
        self.fc2 = nn.Linear(hidden_size, hidden_size).to('cuda:0')
        self.fc3 = nn.Linear(hidden_size, output_size).to('cuda:0')

        # Initialize other parameters and move them to the device
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.weights = torch.from_numpy(np.random.uniform(-1, 1, (num_neurons, num_neurons))).float().to('cuda:0')
        self.last_spike_time = torch.zeros(num_neurons, device='cuda:0')
        self.current_time = 0
        self.membrane_potential = torch.zeros(num_neurons, device='cuda:0')
        self.membrane_decay = 0.5

        # Hodgkin-Huxley parameters initialization
        self.n = torch.zeros(num_neurons, device='cuda:0')
        self.m = torch.zeros(num_neurons, device='cuda:0')
        self.h = torch.zeros(num_neurons, device='cuda:0')
        self.I_Na = None
        self.I_K = None
        self.I_L = None

        # Adjusting Hodgkin-Huxley parameters for specific use case
        self.g_Na = 120.0  # mS/cm^2
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0  # mV
        self.E_K = -77.0
        self.E_L = -54.4
        
        
    def forward(self, x):
        print("Debug: Forward - Input shape: {}".format(x.shape))
        x = self.update_hodgkin_huxley_dynamics(x)
        logging.info("Debug: After update_hodgkin_huxley_dynamics - x shape: %s", x.shape)  # After dynamics update

        x = torch.sigmoid(self.fc2(x))
        logging.info("Debug: After fc2 - x shape: %s", x.shape)  # After fc2

        x = torch.sigmoid(self.fc3(x))
        logging.info("Debug: After fc3 - x shape: %s", x.shape)  # After fc3
        return x


    def update_hodgkin_huxley_dynamics(self, input_signal):
        logging.info("SYSTEM MESSAGE: Inside update_hodgkin_huxley_dynamics")
        # Convert input_signal to a tensor if it's not already
        if not isinstance(input_signal, torch.Tensor):
            input_signal = torch.tensor(input_signal, dtype=torch.float32, device=self.device)



        # Ensure input_signal is two-dimensional [batch_size, feature_size]
        input_signal = input_signal.to(self.device).float()  # Ensure full precision

        # Ensure input_signal is two-dimensional [batch_size, feature_size]
        if len(input_signal.shape) == 1:
            input_signal = input_signal.unsqueeze(0)

        # Hodgkin-Huxley model dynamics
        dt = 0.01

            # Update gating variables
        # Update gating variables
    # Reshape self.n, self.m, self.h to match the gating variables dimensions
        n_reshaped = self.n.unsqueeze(0)
        m_reshaped = self.m.unsqueeze(0)
        h_reshaped = self.h.unsqueeze(0)

        # Update gating variables
        alpha_n = 0.01 * (self.membrane_potential + 55) / (1 - torch.exp(-(self.membrane_potential + 55) / 10)).squeeze(0)
        beta_n = 0.125 * torch.exp(-(self.membrane_potential + 65) / 80).squeeze(0)
        alpha_m = 0.1 * (self.membrane_potential + 40) / (1 - torch.exp(-(self.membrane_potential + 40) / 10)).squeeze(0)
        beta_m = 4.0 * torch.exp(-(self.membrane_potential + 65) / 18).squeeze(0)
        alpha_h = 0.07 * torch.exp(-(self.membrane_potential + 65) / 20).squeeze(0)
        beta_h = 1 / (1 + torch.exp(-(self.membrane_potential + 35) / 10)).squeeze(0)

        # Perform updates using reshaped variables
        self.n = (n_reshaped + dt * (alpha_n * (1 - n_reshaped) - beta_n * n_reshaped)).squeeze(0)
        self.m = (m_reshaped + dt * (alpha_m * (1 - m_reshaped) - beta_m * m_reshaped)).squeeze(0)
        self.h = (h_reshaped + dt * (alpha_h * (1 - h_reshaped) - beta_h * h_reshaped)).squeeze(0)

        self.I_Na = self.g_Na * self.m**3 * self.h * (self.membrane_potential - self.E_Na)
        self.I_K = self.g_K * self.n**4 * (self.membrane_potential - self.E_K)
        self.I_L = self.g_L * (self.membrane_potential - self.E_L)

        # CHANGE HERE WHEN CHANGING THE NUMBER OF NEURONS
        input_signal = input_signal.view(1, 1000)

        # Now the input can be passed through the linear layer
        self.membrane_potential = self.membrane_decay * self.membrane_potential + torch.sigmoid(self.fc1(input_signal))


        spiking_neurons = self.membrane_potential > 1.0
        self.membrane_potential[spiking_neurons] = 0
        logging.info("SYSTEM MESSAGE: Exiting update_hodgkin_huxley_dynamics")
        return self.membrane_potential

    def update_weights(self, action, reward):
        # STDP parameters tuning
        logging.info("SYSTEM MESSAGE: Inside SSN update_weights")
        tau_plus = 15.0
        tau_minus = 25.0
        A_plus = 0.02
        A_minus = 0.015

        # Since self.weights is initially a NumPy array, ensure it is converted to a tensor
        if not isinstance(self.weights, torch.Tensor):
            self.weights = torch.from_numpy(self.weights).float().to('cuda:0').half()

        # Since self.last_spike_time is already a tensor, use it directly
        time_since_last_spike = self.current_time - self.last_spike_time

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if self.is_connection_active(i, j, action):
                    delta_w = A_plus * torch.exp(-torch.abs(time_since_last_spike[i] - time_since_last_spike[j]) / tau_plus) if time_since_last_spike[i] < time_since_last_spike[j] else -A_minus * torch.exp(-torch.abs(time_since_last_spike[i] - time_since_last_spike[j]) / tau_minus)
                    self.weights[i, j] += reward * delta_w

        # Clamp the updated weights to keep them within the [-1, 1] range
        self.weights = torch.clamp(self.weights, -1, 1)

        # Update last_spike_time for the action neuron
        self.last_spike_time[action] = self.current_time
        self.current_time += 1
        logging.info("SYSTEM MESSAGE: Exiting SSN update_weights")



    def get_feedback(self):
        """
        Generate feedback based on the current state of the synaptic weights.
        This feedback can be used to adjust other parts of the AGI system.
        """
        # Calculate the feedback based on the variability and mean strength of the synapses
        # Enhanced feedback mechanism
        weights_tensor = self.weights  # Use the tensor that's already on the GPU
        weight_variability = torch.var(weights_tensor)
        mean_weight_strength = torch.mean(torch.abs(weights_tensor))
        firing_rate = self.membrane_potential.mean().item()  # Average firing rate

        feedback = weight_variability + mean_weight_strength + firing_rate
        logging.info(f"SYSTEM MESSAGE: SNN Feedback: {feedback}")
        return feedback


    def is_connection_active(self, pre_neuron, post_neuron, action):
        """
        Determines if a connection is part of the active pathway based on the current action.
        This version uses a more complex heuristic to determine active connections.
        """
        # Assuming each action activates a set of neurons, not just one.
        active_neurons = self.action_to_active_neurons(action)

        # Check if the connection is between any of the active neurons
        return pre_neuron in active_neurons and post_neuron in active_neurons

    def action_to_active_neurons(self, action):
        """
        Maps an action to a set of active neurons.
        This mapping can be based on a predefined scheme or learned dynamically.
        """
        # Example: map action to neurons based on a predefined pattern or a learned representation
        # This could be a neural network or any other complex mapping mechanism
        # For demonstration, using a simple modular arithmetic approach
        base_neuron = action % self.num_neurons
        return {base_neuron, (base_neuron + 1) % self.num_neurons, (base_neuron - 1) % self.num_neurons}

def train_step(model, images, labels, optimizer, criterion):
    # Here you would write your training logic.
    # For now, it's a placeholder that returns random tensors for demonstration.
    outputs = torch.randn(64, 10)  # Assuming batch size 64 and 10 classes
    loss = torch.tensor([0.0])  # Dummy loss
    return outputs, loss
        
        
        
# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize the SNN
num_neurons = 1000
input_size = 1000  # Size of MNIST images
hidden_size = 1000  # Size of hidden layer
output_size = 100  # Number of classes in MNIST


# Define the loss function and optimizer
# Initialize the SNN and load the model weights only once
snn = SpikingNeuralNetwork(num_neurons, input_size, output_size, hidden_size)
snn.load_state_dict(torch.load('snn_modelV5.pth', map_location=device))
snn.eval()  # Set to evaluation mode

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(snn.parameters(), lr=0.01)

            
    
@app.route('/')
def index():
    return render_template('index.html')  # Serve the main page



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


if __name__ == '__main__':
    app.run(debug=True)


    
