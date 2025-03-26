import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class PDEVisualization:
    @staticmethod
    def spectral_analysis(model, x_range, t_range):
        """
        Perform spectral analysis of the PDE solution
        
        Args:
            model (nn.Module): Trained neural network model
            x_range (torch.Tensor): Spatial coordinates
            t_range (torch.Tensor): Temporal coordinates
        """
        # Create mesh grid
        x_grid, t_grid = torch.meshgrid(x_range.squeeze(), t_range.squeeze())
        coordinates = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        
        # Compute solution
        with torch.no_grad():
            solution = model(coordinates).numpy().reshape(x_grid.shape)
        
        # Compute 2D Fourier Transform
        f_transform = np.fft.fft2(solution)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Original Solution
        plt.subplot(131)
        plt.pcolormesh(x_grid, t_grid, solution, cmap='viridis')
        plt.title('Original Solution')
        plt.xlabel('Spatial Coordinate')
        plt.ylabel('Temporal Coordinate')
        plt.colorbar()
        
        # Magnitude Spectrum
        plt.subplot(132)
        plt.pcolormesh(magnitude_spectrum, cmap='plasma')
        plt.title('Magnitude Spectrum (Log Scale)')
        plt.xlabel('Frequency X')
        plt.ylabel('Frequency T')
        plt.colorbar()
        
        # Phase Spectrum
        plt.subplot(133)
        phase_spectrum = np.angle(f_shift)
        plt.pcolormesh(phase_spectrum, cmap='twilight')
        plt.title('Phase Spectrum')
        plt.xlabel('Frequency X')
        plt.ylabel('Frequency T')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def error_distribution(model, true_solution_func, x_range, t_range):
        """
        Analyze error distribution between model and true solution
        
        Args:
            model (nn.Module): Trained neural network model
            true_solution_func (callable): Analytical solution function
            x_range (torch.Tensor): Spatial coordinates
            t_range (torch.Tensor): Temporal coordinates
        """
        # Create mesh grid
        x_grid, t_grid = torch.meshgrid(x_range.squeeze(), t_range.squeeze())
        coordinates = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        
        # Compute model solution
        with torch.no_grad():
            model_solution = model(coordinates).numpy().reshape(x_grid.shape)
        
        # Compute true solution
        true_solution = np.array([
            true_solution_func(x, t) 
            for x, t in zip(x_grid.flatten(), t_grid.flatten())
        ]).reshape(x_grid.shape)
        
        # Compute error
        error = model_solution - true_solution
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Error Heatmap
        plt.subplot(131)
        plt.pcolormesh(x_grid, t_grid, error, cmap='coolwarm')
        plt.title('Error Heatmap')
        plt.xlabel('Spatial Coordinate')
        plt.ylabel('Temporal Coordinate')
        plt.colorbar()
        
        # Error Distribution
        plt.subplot(132)
        sns.histplot(error.flatten(), kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Frequency')
        
        # Q-Q Plot
        plt.subplot(133)
        stats.probplot(error.flatten(), plot=plt)
        plt.title('Q-Q Plot of Errors')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def sensitivity_analysis(model, x_range, t_range, noise_levels=[0, 0.01, 0.1]):
        """
        Perform sensitivity analysis with different input noise levels
        
        Args:
            model (nn.Module): Trained neural network model
            x_range (torch.Tensor): Spatial coordinates
            t_range (torch.Tensor): Temporal coordinates
            noise_levels (list): Different noise levels to test
        """
        plt.figure(figsize=(15, 5))
        
        for i, noise_level in enumerate(noise_levels):
            # Create mesh grid with noise
            x_grid, t_grid = torch.meshgrid(x_range.squeeze(), t_range.squeeze())
            
            # Add Gaussian noise
            x_noisy = x_grid + np.random.normal(0, noise_level, x_grid.shape)
            t_noisy = t_grid + np.random.normal(0, noise_level, t_grid.shape)
            
            # Combine coordinates
            coordinates = torch.stack([x_noisy.flatten(), t_noisy.flatten()], dim=1)
            
            # Compute solution
            with torch.no_grad():
                solution = model(coordinates).numpy().reshape(x_grid.shape)
            
            # Plot
            plt.subplot(1, len(noise_levels), i+1)
            plt.pcolormesh(x_grid, t_grid, solution, cmap='viridis')
            plt.title(f'Noise Level: {noise_level}')
            plt.xlabel('Spatial Coordinate')
            plt.ylabel('Temporal Coordinate')
            plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def derivative_analysis(model, x_range, t_range):
        """
        Analyze spatial and temporal derivatives of the solution
        
        Args:
            model (nn.Module): Trained neural network model
            x_range (torch.Tensor): Spatial coordinates
            t_range (torch.Tensor): Temporal coordinates
        """
        # Create mesh grid
        x_grid, t_grid = torch.meshgrid(x_range.squeeze(), t_range.squeeze())
        coordinates = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        coordinates.requires_grad_(True)
        
        # Compute solution
        solution = model(coordinates)
        
        # Compute derivatives
        grad = torch.autograd.grad(
            solution, coordinates, 
            grad_outputs=torch.ones_like(solution),
            create_graph=True
        )[0]
        
        # Reshape derivatives
        du_dx = grad[:, 0].detach().numpy().reshape(x_grid.shape)
        du_dt = grad[:, 1].detach().numpy().reshape(x_grid.shape)
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Original Solution
        plt.subplot(131)
        plt.pcolormesh(x_grid, t_grid, solution.detach().numpy().reshape(x_grid.shape), cmap='viridis')
        plt.title('Original Solution')
        plt.colorbar()
        
        # Spatial Derivative
        plt.subplot(132)
        plt.pcolormesh(x_grid, t_grid, du_dx, cmap='coolwarm')
        plt.title('Spatial Derivative (du/dx)')
        plt.colorbar()
        
        # Temporal Derivative
        plt.subplot(133)
        plt.pcolormesh(x_grid, t_grid, du_dt, cmap='coolwarm')
        plt.title('Temporal Derivative (du/dt)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

class HarmonicPDESolver:
    def __init__(self, pde_type='heat', domain_config=None):
        """
        Initialize the Harmonic Domain PDE Solver
        
        Args:
            pde_type (str): Type of PDE to solve ('heat', 'wave', 'navier_stokes')
            domain_config (dict): Configuration for the physical domain
        """
        self.pde_type = pde_type
        self.default_config = {
            'heat': {
                'x_min': 0, 'x_max': 1,
                't_min': 0, 't_max': 1,
                'diffusivity': 0.01
            },
            'wave': {
                'x_min': 0, 'x_max': np.pi,
                't_min': 0, 't_max': 2,
                'wave_speed': 1.0
            }
        }
        
        self.domain_config = domain_config or self.default_config[pde_type]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    class HarmonicNeuralNetwork(nn.Module):
        def __init__(self, input_dim=2, hidden_layers=[50, 50], 
                     harmonic_features=64):
            """
            Harmonic Neural Network Architecture
            
            Args:
                input_dim (int): Dimension of input (typically 2 for x,t)
                hidden_layers (list): Neurons in hidden layers
                harmonic_features (int): Number of harmonic features
            """
            super().__init__()
            
            # Fourier Feature Mapping
            self.harmonic_mapping = nn.Sequential(
                nn.Linear(input_dim, harmonic_features),
                nn.ReLU(),
                nn.Linear(harmonic_features, harmonic_features)
            )
            
            # Main network layers
            layers = []
            layer_sizes = [harmonic_features] + hidden_layers + [1]
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            """
            Forward pass through harmonic network
            
            Args:
                x (torch.Tensor): Input coordinates
            
            Returns:
                torch.Tensor: PDE solution at input coordinates
            """
            harmonic_input = self.harmonic_mapping(x)
            return self.network(harmonic_input)
    
    def physics_loss(self, model, x, t):
        """
        Compute physics-informed loss based on PDE residual
        
        Args:
            model (nn.Module): Neural network model
            x (torch.Tensor): Spatial coordinates
            t (torch.Tensor): Temporal coordinates
        
        Returns:
            torch.Tensor: Physics loss
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        # Combine coordinates
        coordinates = torch.cat([x, t], dim=1)
        u = model(coordinates)
        
        # Compute derivatives
        grad_u = torch.autograd.grad(
            u, coordinates, 
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        du_dx = grad_u[:, 0:1]
        du_dt = grad_u[:, 1:2]
        
        # Second derivative
        hessian = torch.autograd.grad(
            du_dx, x, 
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0]
        
        d2u_dx2 = hessian[:, 0:1]
        
        # PDE Residual (Heat Equation)
        if self.pde_type == 'heat':
            diff = self.domain_config['diffusivity']
            pde_residual = du_dt - diff * d2u_dx2
        elif self.pde_type == 'wave':
            wave_speed = self.domain_config['wave_speed']
            hessian_t = torch.autograd.grad(
                du_dt, t, 
                grad_outputs=torch.ones_like(du_dt),
                create_graph=True
            )[0]
            d2u_dt2 = hessian_t[:, 0:1]
            pde_residual = d2u_dt2 - (wave_speed**2 * d2u_dx2)
        
        return torch.mean(pde_residual**2)
    
    def train(self, epochs=5000, learning_rate=1e-3):
        """
        Train the Harmonic PDE Solver
        
        Args:
            epochs (int): Number of training epochs
            learning_rate (float): Optimization learning rate
        """
        # Generate training points
        x = torch.linspace(
            self.domain_config['x_min'], 
            self.domain_config['x_max'], 
            100
        ).reshape(-1, 1)
        t = torch.linspace(
            self.domain_config['t_min'], 
            self.domain_config['t_max'], 
            100
        ).reshape(-1, 1)
        
        # Mesh grid for coordinates
        x_grid, t_grid = torch.meshgrid(x.squeeze(), t.squeeze())
        coordinates = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        
        # Initialize model and optimizer
        model = self.HarmonicNeuralNetwork().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.physics_loss(model, coordinates[:, 0:1], coordinates[:, 1:2])
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        return model
    
    def visualize_solution(self, model):
        """
        Visualize PDE solution
        
        Args:
            model (nn.Module): Trained neural network model
        """
        x = torch.linspace(
            self.domain_config['x_min'], 
            self.domain_config['x_max'], 
            100
        ).reshape(-1, 1)
        t = torch.linspace(
            self.domain_config['t_min'], 
            self.domain_config['t_max'], 
            100
        ).reshape(-1, 1)
        
        x_grid, t_grid = torch.meshgrid(x.squeeze(), t.squeeze())
        coordinates = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        
        with torch.no_grad():
            solution = model(coordinates).numpy().reshape(x_grid.shape)
        
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(x_grid, t_grid, solution, cmap='viridis')
        plt.colorbar(label='Solution')
        plt.title(f'{self.pde_type.capitalize()} Equation Solution')
        plt.xlabel('Spatial Coordinate')
        plt.ylabel('Temporal Coordinate')
        plt.show()

def main():
    # Initialize and train solver for heat equation
    solver = HarmonicPDESolver(pde_type='heat')
    trained_model = solver.train()

     # Prepare coordinate ranges
    x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_range = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    # Perform various visualizations
    # 1. Spectral Analysis
    PDEVisualization.spectral_analysis(trained_model, x_range, t_range)
    
    def true_solution(x, t):
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    PDEVisualization.error_distribution(trained_model, true_solution, x_range, t_range)
    
    # 3. Sensitivity Analysis
    PDEVisualization.sensitivity_analysis(trained_model, x_range, t_range)
    
    # 4. Derivative Analysis
    PDEVisualization.derivative_analysis(trained_model, x_range, t_range)
    
    # Visualize solution
    # solver.visualize_solution(trained_model)

if __name__ == '__main__':
    main()
