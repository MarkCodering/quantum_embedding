import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
import pennylane as qml
from pennylane.templates import BasicEntanglerLayers
from pennylane import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
learning_rate = 3e-4
num_epochs = 10

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create a quantum neural network in Pennylane and wrap it in a PyTorch module
# Set a random seed
np.random.seed(42)

# Define the quantum circuit
n_qubits = 10
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_neural_network(inputs, weights):
    # Encoding layer
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Variational layer
    BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Measurement layer
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Define the quantum neural network class
class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_layers=2):
        super(QuantumNeuralNetwork, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_weights = n_layers * n_qubits
        self.weights = nn.Parameter(2 * np.pi * torch.rand(self.n_layers, self.n_qubits))
    
    def forward(self, x):
        # Apply the quantum neural network to each element of the input batch
        out = torch.stack([torch.tensor(quantum_neural_network(x[i], self.weights), dtype=torch.float32) for i in range(len(x))])
        return out
    
# Define the hybrid quantum-classical neural network
class HybridModel(nn.Module):
    def __init__(self, n_qubits):
        super(HybridModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.classical = nn.Linear(1000, n_qubits)
        self.quantum = QuantumNeuralNetwork(n_qubits)
        self.output = nn.Linear(n_qubits, 10)
    
    def forward(self, x):
        x = self.vit(x).logits
        x = self.classical(x)
        x = self.quantum(x)
        x = self.output(x)
        
        return x

model = HybridModel(n_qubits).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# Evaluate the model
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Save the model checkpoint
torch.save(model.state_dict(), 'vit_qnn_cifar10.pth')