# Harmonic PDE Solver

## Overview

This project implements a Physics-Informed Neural Network (PINN) for solving Partial Differential Equations (PDEs) using advanced harmonic feature mapping and neural network architectures. The implementation focuses on solving Heat and Wave equations with innovative visualization and analysis techniques.

## Features

- 🧮 Physics-Informed Neural Network (PINN) solver
- 🌊 Supports multiple PDE types (Heat and Wave equations)
- 📊 Comprehensive visualization and analysis methods
- 🔬 Spectral and derivative analysis
- 📈 Error distribution and sensitivity testing

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- DeepXDE
- Matplotlib
- Seaborn
- SciPy

## Installation

```bash
# Clone the repository
git clone https://github.com/barishandas/Harmonic-Neural-Network.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Solving Example

```python
# Initialize solver for heat equation
solver = HarmonicPDESolver(pde_type='heat')
trained_model = solver.train()

# Visualize solution
solver.visualize_solution(trained_model)
```

### Advanced Analysis

```python
# Prepare coordinate ranges
x_range = torch.linspace(0, 1, 100).reshape(-1, 1)
t_range = torch.linspace(0, 1, 100).reshape(-1, 1)

# Perform spectral analysis
PDEVisualization.spectral_analysis(trained_model, x_range, t_range)

# Perform error distribution analysis
def true_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

PDEVisualization.error_distribution(trained_model, true_solution, x_range, t_range)
```

## Key Components

### `HarmonicPDESolver`
- Solves PDEs using a novel neural network architecture
- Supports Heat and Wave equations
- Implements physics-informed loss function

### `PDEVisualization`
- Spectral analysis of PDE solutions
- Error distribution visualization
- Sensitivity analysis
- Derivative analysis

## Visualization Techniques

1. **Spectral Analysis**: Fourier transform of solution
2. **Error Distribution**: Comparison with analytical solutions
3. **Sensitivity Analysis**: Performance under noise
4. **Derivative Analysis**: Spatial and temporal derivatives

## Theoretical Background

The solver uses Physics-Informed Neural Networks (PINNs) with harmonic feature mapping to solve PDEs. The approach combines machine learning techniques with physical constraints to create more accurate and interpretable solutions.

## Supported PDE Types

- Heat Equation
- Wave Equation
- (Extensible to other PDE types)

## Performance Metrics

- Physics-informed loss minimization
- Comparison with analytical solutions
- Computational efficiency
- Robustness to input noise

## Limitations

- Primarily designed for 1D and 2D PDEs
- Performance may vary based on PDE complexity
- Requires careful hyperparameter tuning

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Work

- Extend to higher-dimensional PDEs
- Improve harmonic mapping techniques
- Develop more advanced physics-informed loss functions

## References

1. Raissi, M., et al. (2019). Physics-informed neural networks.
2. Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions.
