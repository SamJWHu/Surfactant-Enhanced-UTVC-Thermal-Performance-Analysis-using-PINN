# Surfactant-Enhanced-UTVC-Thermal-Performance-Analysis-using-PINN

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** using TensorFlow to simulate and analyze the **thermal performance enhancement** of **surfactant-enhanced ultra-thin vapor chambers (UTVC)**. The PINN integrates governing physical equations, including the **continuity equation**, **Navier-Stokes equations**, **energy equation**, and **surfactant transport equation**, to accurately model heat dissipation mechanisms within UTVCs under various surfactant conditions.

## Features

- **3D PINN Architecture:** Designed to handle three-dimensional simulations for comprehensive analysis.
- **Integration of Governing Equations:** Enforces physical laws directly within the neural network's loss function.
- **Detailed Boundary Conditions:** Models evaporation, condensation, adiabatic, and convective boundaries accurately.
- **Normalization and Denormalization:** Ensures numerical stability and efficient training.
- **Visualization Tools:** This tool generates plots for temperature distribution, velocity fields, pressure distribution, and surfactant concentration at various time points.
- **Optimized Training:** Implements strategies to handle high-dimensional data efficiently, including gradient clipping and learning rate scheduling.

## Background

Ultra-thin vapor Chambers (UTVC) are critical components in thermal management systems, particularly in electronics cooling. Enhancing their thermal performance can significantly improve device reliability and efficiency. This project leverages PINNs to model the complex heat transfer and fluid dynamics within UTVCs, incorporating the effects of surfactants to optimize performance.

## Getting Started

### Prerequisites

- **Python 3.8 or higher**
- **TensorFlow 2.8.0 or higher**
- **NumPy**
- **Matplotlib**

Ensure you have Python and `pip` installed on your system. It's recommended to use a virtual environment to manage dependencies.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/pinn-utvc.git
   cd pinn-utvc

## Introduction

Efficient thermal management is critical in modern electronic devices, especially those housing high-performance components like CPUs and GPUs. **Ultra-Thin Vapor Chambers (UTVCs)** are widely used for dissipating heat effectively. However, the thermal performance of UTVCs can be further enhanced by modifying the working fluid properties. This project leverages **Physics-Informed Neural Networks (PINNs)** to simulate and analyze the thermal performance of **Surfactant-Enhanced Ultra-Thin Vapor Chambers**. By integrating various dilute surfactant-water mixtures, the model aims to optimize heat dissipation, ensuring optimal performance and longevity of electronic devices.

## Features

- **3D Physics-Informed Neural Network (PINN):** Extends traditional PINNs to three dimensions, capturing the thickness of the UTVC for accurate thermal modeling.
- **Surfactant Integration:** Evaluates ten different dilute surfactant-water mixtures to assess their impact on surface tension reduction and heat transfer efficiency.
- **Comprehensive Loss Function:** Incorporates governing equations (Continuity, Navier-Stokes, Energy, and Surfactant Transport) into the loss function to ensure physically consistent predictions.
- **Normalization and Scaling:** Applies normalization techniques to inputs and outputs to stabilize training and improve convergence.
- **Training Optimizations:** Utilizes learning rate scheduling, gradient clipping, and a streamlined network architecture to reduce training time and enhance performance.
- **Visualization:** Generates clear visual comparisons of maximum and average temperatures across different surfactant mixtures, facilitating easy identification of optimal mixtures.
- **Efficient Training:** Optimized to run on GPUs, significantly reducing training time from hours to minutes where possible.

## Surfactant Mixtures

The model evaluates the following ten dilute surfactant-water mixtures to determine their effectiveness in enhancing the thermal performance of UTVCs:

1. **Polyethylene Glycol 600 (PEG600):**
   - 0.3 wt%
   - 0.4 wt%
2. **Polyethylene Glycol 1000 (PEG1000):**
   - 0.2 wt%
   - 0.3 wt%
3. **Polypropylene Glycol 600 (PPG600):**
   - 0.2 wt%
   - 0.3 wt%
4. **Polypropylene Glycol 1000 (PPG1000):**
   - 0.1 wt%
   - 0.2 wt%
5. **C12-C13 Alkyl Polyglycosides (C12-C13 APGs):**
   - 0.1 wt%
   - 0.15 wt%

**Surfactant Properties:**

| Surfactant              | Concentration (wt%) | Effective Surface Tension (N/m) | Surface Coverage (%) |
|-------------------------|---------------------|----------------------------------|----------------------|
| PEG600                  | 0.3                 | 55-65                            | 30-40                |
| PEG600                  | 0.4                 | 55-65                            | 30-40                |
| PEG1000                 | 0.2                 | 58-68                            | 20-30                |
| PEG1000                 | 0.3                 | 58-68                            | 20-30                |
| PPG600                  | 0.2                 | 58-68                            | 20-30                |
| PPG600                  | 0.3                 | 58-68                            | 20-30                |
| PPG1000                 | 0.1                 | 60-70                            | 10-20                |
| PPG1000                 | 0.2                 | 60-70                            | 10-20                |
| C12-C13 APGs            | 0.1                 | 30-40                            | 90-100               |
| C12-C13 APGs            | 0.15                | 30-40                            | 90-100               |


Results
Upon training, the PINN predicts the thermal and fluid dynamics within the UTVC. The visualization plots provide insights into temperature distribution, velocity fields, pressure gradients, and surfactant concentrations at various time points, facilitating the analysis of thermal performance enhancements.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/utvc-pinn.git
   cd utvc-pinn

Usage
Configure Surfactant Properties

The surfactant_data dictionary in pinn_model.py contains the properties of each surfactant mixture. Modify this section if you wish to add or update new mixtures.

Run the Simulation

Execute the main script to train the PINN and evaluate thermal performance:

bash
Copy code
python pinn_model.py
View Results

After training, the script will generate bar charts comparing each surfactant mixture's maximum and average temperatures. These visualizations help identify the most effective surfactants for enhancing UTVC performance.

Results
The simulation evaluates the effectiveness of each surfactant mixture in reducing maximum temperatures near the evaporation area and maintaining reasonable average temperatures across the UTVC. The results are visualized through bar charts, clearly comparing thermal performance enhancements.

Project Structure
Copy code
utvc-pinn/
│
├── assets/
│   ├── utvc_simulation.png
│   └── results_comparison.png
│
├── pinn_model.py
├── requirements.txt
├── README.md
└── LICENSE
assets/: Contains images for the README and other documentation.
pinn_model.py: Main script implementing the 3D PINN model.
requirements.txt: Lists all Python dependencies.
README.md: Project documentation (this file).
LICENSE: License information.
Contributing
Contributions are welcome! Your input is valuable, whether it's improving the model, adding new surfactant mixtures, or enhancing documentation.

Fork the Repository

bash
Copy code
git clone https://github.com/yourusername/utvc-pinn.git
cd utvc-pinn
Create a Feature Branch

bash
Copy code
git checkout -b feature/YourFeature
Commit Your Changes

bash
Copy code
git commit -m "Add your message here"
Push to the Branch

bash
Copy code
git push origin feature/YourFeature
Open a Pull Request

License
This project is licensed under the MIT License.

Acknowledgements
Physics-Informed Neural Networks by Raissi, Perdikaris, and Karniadakis.
TensorFlow for providing the deep learning framework.
NumPy and Matplotlib for data handling and visualization.

Contact
For any questions or suggestions, please open an issue or contact r02522318@gmail.com
