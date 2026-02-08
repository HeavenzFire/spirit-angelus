import torch
    import torch.nn as nn
    
    class EvolvingNetwork(nn.Module):
        def __init__(self, input_dim=1321):
            super(EvolvingNetwork, self).__init__()
            self.layers = nn.ModuleList([nn.Linear(input_dim, 512)])
            self.evolution_threshold = 0.85
    
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return x
    
        def needs_expansion(self, complexity_metric):
            return complexity_metric > self.evolution_threshold and len(self.layers) < 12 
    
        def add_resonance_layer(self):
            current_dim = self.layers[-1].out_features
            self.layers.append(nn.Linear(current_dim, current_dim))
