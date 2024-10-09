# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 02:27:19 2024

@author: SamJWHu
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Physical constants (SI units)
rho = 998          # Density of water at ~30°C (kg/m³)
mu = 1.0e-3        # Dynamic viscosity of water (Pa·s)
cp = 4182          # Specific heat capacity of water (J/(kg·K))
k = 0.6            # Thermal conductivity of water (W/(m·K))
D = 1.0e-9         # Diffusion coefficient of surfactant in water (m²/s)
sigma_water = 0.0728  # Surface tension of water (N/m)
h_conv = 1000       # Convective heat transfer coefficient (W/(m²·K))

# UTVC dimensions (meters)
L = 0.2           # Length (200 mm)
W = 0.1           # Width (100 mm)
H = 0.0015        # Thickness (1.5 mm)

# Heat sources
Q_CPU = 30        # CPU heat (W)
Q_GPU = 50        # GPU heat (W)

# Ambient temperature
T_ambient = 30    # Ambient temperature in °C


# Characteristic scales
L_star = L        # Length scale (m)
U_star = 0.01     # Velocity scale (m/s), estimated
T_star = 40       # Temperature difference scale (K), from ambient to maximum expected temperature
C_star = 0.003    # Concentration scale (kg/kg), 0.3 wt% PEG600
t_star = L_star / U_star  # Characteristic time (s)


# Reynolds number
Re = (rho * U_star * L_star) / mu
print(f"Reynolds number (Re): {Re:.2f}")

# Prandtl number
Pr = (mu * cp) / k
print(f"Prandtl number (Pr): {Pr:.2f}")

# Péclet number for heat
Pe_T = Re * Pr
print(f"Péclet number for heat (Pe_T): {Pe_T:.2e}")

# Péclet number for concentration
Pe_C = (U_star * L_star) / D
print(f"Péclet number for concentration (Pe_C): {Pe_C:.2e}")

# Dimensionless convective heat transfer coefficient
h_dim = (h_conv * L_star) / k
print(f"Dimensionless convective heat transfer coefficient (h_dim): {h_dim:.2f}")



def normalize_x(x):
    return x / L_star

def normalize_y(y):
    return y / W

def normalize_z(z):
    return z / H

def normalize_t(t):
    return t / t_star

def normalize_u(u):
    return u / U_star

def normalize_v(v):
    return v / U_star

def normalize_w(w):
    return w / U_star

def normalize_T(T):
    return (T - T_ambient) / T_star

def normalize_C(C):
    return C / C_star



def denormalize_T(T_dimless):
    return T_dimless * T_star + T_ambient

def denormalize_u(u_dimless):
    return u_dimless * U_star

def denormalize_v(v_dimless):
    return v_dimless * U_star

def denormalize_w(w_dimless):
    return w_dimless * U_star



# Neural network hyperparameters
input_dim = 4          # x, y, z, t
output_dim = 6         # u, v, w, p, T, C
num_hidden_layers = 8
num_neurons_per_layer = 50
activation_function = tf.nn.tanh

# Training parameters
learning_rate = 1e-3
epochs = 10000
batch_size = 256



def generate_collocation_points(num_points):
    x = np.random.uniform(0, 1, num_points).astype(np.float32)
    y = np.random.uniform(0, 1, num_points).astype(np.float32)
    z = np.random.uniform(0, 1, num_points).astype(np.float32)
    t = np.random.uniform(0, 1, num_points).astype(np.float32)  # Normalized time
    X_colloc = np.stack([x, y, z, t], axis=1)
    return X_colloc



def is_in_cpu_region(x, y, z):
    return (x >= normalize_x(0.3 * L_star)) & (x <= normalize_x(0.4 * L_star)) & \
           (y >= normalize_y(0.45 * W)) & (y <= normalize_y(0.55 * W)) & \
           (z >= 0) & (z <= 1)

def is_in_gpu_region(x, y, z):
    return (x >= normalize_x(0.6 * L_star)) & (x <= normalize_x(0.7 * L_star)) & \
           (y >= normalize_y(0.45 * W)) & (y <= normalize_y(0.55 * W)) & \
           (z >= 0) & (z <= 1)



def generate_boundary_points(num_points):
    x = np.random.uniform(0, 1, num_points).astype(np.float32)
    y = np.random.uniform(0, 1, num_points).astype(np.float32)
    z = np.random.uniform(0, 1, num_points).astype(np.float32)
    t = np.random.uniform(0, 1, num_points).astype(np.float32)
    
    X = np.stack([x, y, z, t], axis=1)
    
    # CPU region
    cpu_mask = is_in_cpu_region(x, y, z)
    X_evap_cpu = X[cpu_mask]
    
    # GPU region
    gpu_mask = is_in_gpu_region(x, y, z)
    X_evap_gpu = X[gpu_mask]
    
    # Left boundary (x=0)
    left_mask = (x <= 1e-6)
    X_cond_left = X[left_mask]
    
    # Right boundary (x=1)
    right_mask = (x >= 1 - 1e-6)
    X_cond_right = X[right_mask]
    
    # Top and bottom boundaries (y=0 and y=1) - Adiabatic
    top_mask = (y >= 1 - 1e-6)
    bottom_mask = (y <= 1e-6)
    X_adiabatic = np.concatenate((X[top_mask], X[bottom_mask]), axis=0)
    
    # Combine boundary points
    X_boundary = {
        'evap_cpu': X_evap_cpu,
        'evap_gpu': X_evap_gpu,
        'cond_left': X_cond_left,
        'cond_right': X_cond_right,
        'adiabatic': X_adiabatic
    }
    
    return X_boundary



def generate_initial_points(num_points):
    x = np.random.uniform(0, 1, num_points).astype(np.float32)
    y = np.random.uniform(0, 1, num_points).astype(np.float32)
    z = np.random.uniform(0, 1, num_points).astype(np.float32)
    t = np.zeros(num_points, dtype=np.float32)
    X_init = np.stack([x, y, z, t], axis=1)
    return X_init



class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(num_neurons_per_layer,
                                                            activation=activation_function,
                                                            kernel_initializer='glorot_normal'))
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=None)  # Outputs: u, v, w, p, T, C

    def call(self, X):
        Z = X
        for layer in self.hidden_layers:
            Z = layer(Z)
        output = self.output_layer(Z)
        return output



def compute_pde_loss(model, X):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            output = model(X)
            u, v, w, p, T, C = tf.split(output, num_or_size_splits=6, axis=1)
        
        # First-order derivatives
        u_x = tape1.gradient(u, X)[:, 0:1]
        u_y = tape1.gradient(u, X)[:, 1:2]
        u_z = tape1.gradient(u, X)[:, 2:3]
        u_t = tape1.gradient(u, X)[:, 3:4]
        
        v_x = tape1.gradient(v, X)[:, 0:1]
        v_y = tape1.gradient(v, X)[:, 1:2]
        v_z = tape1.gradient(v, X)[:, 2:3]
        v_t = tape1.gradient(v, X)[:, 3:4]
        
        w_x = tape1.gradient(w, X)[:, 0:1]
        w_y = tape1.gradient(w, X)[:, 1:2]
        w_z = tape1.gradient(w, X)[:, 2:3]
        w_t = tape1.gradient(w, X)[:, 3:4]
        
        T_x = tape1.gradient(T, X)[:, 0:1]
        T_y = tape1.gradient(T, X)[:, 1:2]
        T_z = tape1.gradient(T, X)[:, 2:3]
        T_t = tape1.gradient(T, X)[:, 3:4]
        
        C_x = tape1.gradient(C, X)[:, 0:1]
        C_y = tape1.gradient(C, X)[:, 1:2]
        C_z = tape1.gradient(C, X)[:, 2:3]
        C_t = tape1.gradient(C, X)[:, 3:4]
        
        p_x = tape1.gradient(p, X)[:, 0:1]
        p_y = tape1.gradient(p, X)[:, 1:2]
        p_z = tape1.gradient(p, X)[:, 2:3]
    
    # Second-order derivatives
    u_xx = tape2.gradient(u_x, X)[:, 0:1]
    u_yy = tape2.gradient(u_y, X)[:, 1:2]
    u_zz = tape2.gradient(u_z, X)[:, 2:3]
    
    v_xx = tape2.gradient(v_x, X)[:, 0:1]
    v_yy = tape2.gradient(v_y, X)[:, 1:2]
    v_zz = tape2.gradient(v_z, X)[:, 2:3]
    
    w_xx = tape2.gradient(w_x, X)[:, 0:1]
    w_yy = tape2.gradient(w_y, X)[:, 1:2]
    w_zz = tape2.gradient(w_z, X)[:, 2:3]
    
    T_xx = tape2.gradient(T_x, X)[:, 0:1]
    T_yy = tape2.gradient(T_y, X)[:, 1:2]
    T_zz = tape2.gradient(T_z, X)[:, 2:3]
    
    C_xx = tape2.gradient(C_x, X)[:, 0:1]
    C_yy = tape2.gradient(C_y, X)[:, 1:2]
    C_zz = tape2.gradient(C_z, X)[:, 2:3]
    
    del tape1
    del tape2
    
    # Continuity equation residual
    continuity = u_x + v_y + w_z
    
    # Navier-Stokes equations residuals (dimensionless form)
    momentum_u = Re * (u_t + u * u_x + v * u_y + w * u_z) + p_x - (u_xx + u_yy + u_zz)
    momentum_v = Re * (v_t + u * v_x + v * v_y + w * v_z) + p_y - (v_xx + v_yy + v_zz)
    momentum_w = Re * (w_t + u * w_x + v * w_y + w * w_z) + p_z - (w_xx + w_yy + w_zz)
    
    # Energy equation residual
    energy = Pe_T * (T_t + u * T_x + v * T_y + w * T_z) - (T_xx + T_yy + T_zz)
    
    # Surfactant transport equation residual
    surfactant = Pe_C * (C_t + u * C_x + v * C_y + w * C_z) - (C_xx + C_yy + C_zz)
    
    # Mean squared errors
    mse_continuity = tf.reduce_mean(tf.square(continuity))
    mse_momentum_u = tf.reduce_mean(tf.square(momentum_u))
    mse_momentum_v = tf.reduce_mean(tf.square(momentum_v))
    mse_momentum_w = tf.reduce_mean(tf.square(momentum_w))
    mse_energy = tf.reduce_mean(tf.square(energy))
    mse_surfactant = tf.reduce_mean(tf.square(surfactant))
    
    total_pde_loss = mse_continuity + mse_momentum_u + mse_momentum_v + mse_momentum_w + mse_energy + mse_surfactant
    
    return total_pde_loss



def compute_boundary_loss(model, X_boundary):
    total_bc_loss = 0.0
    
    # Evaporation regions (CPU and GPU)
    for region in ['evap_cpu', 'evap_gpu']:
        X_evap = tf.convert_to_tensor(X_boundary[region], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X_evap)
            output = model(X_evap)
            T = output[:, 4:5]
        T_n = tape.gradient(T, X_evap)[:, 2:3]  # Assuming normal in z-direction (vertical)
        
        # Heat flux boundary condition
        if region == 'evap_cpu':
            q_flux = Q_CPU / (k * T_star / L_star)
        else:
            q_flux = Q_GPU / (k * T_star / L_star)
        
        q_flux_dim = q_flux / (k * T_star / L_star)
        
        mse_q = tf.reduce_mean(tf.square(-T_n - q_flux_dim))
        total_bc_loss += mse_q
    
    # Condensation regions (Left and Right boundaries)
    for side in ['cond_left', 'cond_right']:
        X_cond = tf.convert_to_tensor(X_boundary[side], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X_cond)
            output = model(X_cond)
            T = output[:, 4:5]
        T_n = tape.gradient(T, X_cond)[:, 0:1]  # Normal in x-direction
        
        # Convection boundary condition: -k * dT/dn = h_conv * (T - T_inf)
        T_inf_norm = normalize_T(T_ambient)
        h_dimless = h_conv * L_star / k
        q_conv = h_dimless * (T - T_inf_norm)
        
        mse_conv = tf.reduce_mean(tf.square(-T_n - q_conv))
        total_bc_loss += mse_conv
    
    # Adiabatic boundaries (Top and Bottom)
    X_adiabatic = tf.convert_to_tensor(X_boundary['adiabatic'], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_adiabatic)
        output = model(X_adiabatic)
        T = output[:, 4:5]
    T_n = tape.gradient(T, X_adiabatic)[:, 1:2]  # Normal in y-direction
    
    mse_adiabatic = tf.reduce_mean(tf.square(T_n))
    total_bc_loss += mse_adiabatic
    
    return total_bc_loss



def compute_initial_loss(model, X_init):
    output = model(X_init)
    u_pred, v_pred, w_pred, p_pred, T_pred, C_pred = tf.split(output, num_or_size_splits=6, axis=1)
    
    # Initial conditions (normalized)
    u_init = tf.zeros_like(u_pred)
    v_init = tf.zeros_like(v_pred)
    w_init = tf.zeros_like(w_pred)
    T_init = tf.zeros_like(T_pred)  # T' = 0 corresponds to ambient temperature
    C_init = tf.ones_like(C_pred)   # C' = 1 corresponds to initial concentration
    
    mse_u = tf.reduce_mean(tf.square(u_pred - u_init))
    mse_v = tf.reduce_mean(tf.square(v_pred - v_init))
    mse_w = tf.reduce_mean(tf.square(w_pred - w_init))
    mse_T = tf.reduce_mean(tf.square(T_pred - T_init))
    mse_C = tf.reduce_mean(tf.square(C_pred - C_init))
    
    total_ic_loss = mse_u + mse_v + mse_w + mse_T + mse_C
    
    return total_ic_loss



def loss_function(model, X_col, X_boundary, X_init):
    loss_pde = compute_pde_loss(model, X_col)
    loss_bc = compute_boundary_loss(model, X_boundary)
    loss_ic = compute_initial_loss(model, X_init)
    
    total_loss = loss_pde + loss_bc + loss_ic
    return total_loss, loss_pde, loss_bc, loss_ic



optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)



@tf.function
def train_step(model, optimizer, X_col, X_boundary, X_init):
    with tf.GradientTape() as tape:
        total_loss, loss_pde, loss_bc, loss_ic = loss_function(model, X_col, X_boundary, X_init)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, loss_pde, loss_bc, loss_ic



def train_model(model, optimizer, X_col, X_boundary, X_init):
    loss_history = []
    pde_loss_history = []
    bc_loss_history = []
    ic_loss_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss_value, loss_pde_value, loss_bc_value, loss_ic_value = train_step(
            model, optimizer, X_col, X_boundary, X_init)
        
        loss_history.append(total_loss_value.numpy())
        pde_loss_history.append(loss_pde_value.numpy())
        bc_loss_history.append(loss_bc_value.numpy())
        ic_loss_history.append(loss_ic_value.numpy())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss_value.numpy():.4e}, "
                  f"PDE Loss: {loss_pde_value.numpy():.4e}, "
                  f"Boundary Loss: {loss_bc_value.numpy():.4e}, "
                  f"Initial Loss: {loss_ic_value.numpy():.4e}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Plot training loss
    plt.figure(figsize=(10,6))
    plt.semilogy(loss_history, label='Total Loss')
    plt.semilogy(pde_loss_history, label='PDE Loss')
    plt.semilogy(bc_loss_history, label='Boundary Loss')
    plt.semilogy(ic_loss_history, label='Initial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss History')
    plt.grid(True)
    plt.show()
    
    return loss_history, pde_loss_history, bc_loss_history, ic_loss_history



def evaluate_and_visualize(model, times):
    N_test = 50  # Adjust for higher resolution if needed
    x = np.linspace(0, 1, N_test)
    y = np.linspace(0, 1, N_test)
    z = np.linspace(0, 1, N_test)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    for t_real in times:
        t_norm = normalize_t(t_real)
        T_test = t_norm * np.ones_like(X.flatten())
        X_test = np.stack([X.flatten(), Y.flatten(), Z.flatten(), T_test], axis=1)
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        
        output_pred = model(X_test_tf)
        u_pred, v_pred, w_pred, p_pred, T_pred, C_pred = tf.split(output_pred, num_or_size_splits=6, axis=1)
        
        # Denormalize
        u_pred_dim = denormalize_u(u_pred.numpy()).reshape(N_test, N_test, N_test)
        v_pred_dim = denormalize_v(v_pred.numpy()).reshape(N_test, N_test, N_test)
        w_pred_dim = denormalize_w(w_pred.numpy()).reshape(N_test, N_test, N_test)
        T_pred_dim = denormalize_T(T_pred.numpy()).reshape(N_test, N_test, N_test)
        p_pred_dim = p_pred.numpy().reshape(N_test, N_test, N_test)  # Pressure remains dimensionless
        C_pred_dim = C_pred.numpy().reshape(N_test, N_test, N_test) * C_star
        
        # Visualization at mid-plane z = 0.5
        idx_z = N_test // 2
        
        # Temperature Distribution
        plt.figure(figsize=(8,6))
        plt.contourf(X[:,:,idx_z]*L_star*1000, Y[:,:,idx_z]*W*1000, T_pred_dim[:,:,idx_z], levels=50, cmap='jet')
        plt.colorbar(label='Temperature (°C)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(f'Temperature Distribution at t = {t_real} s')
        plt.show()
        
        # Velocity Field
        plt.figure(figsize=(8,6))
        skip = (slice(None, None, 5), slice(None, None, 5))
        plt.quiver(X[:,:,idx_z][skip]*L_star*1000, Y[:,:,idx_z][skip]*W*1000,
                   u_pred_dim[:,:,idx_z][skip], v_pred_dim[:,:,idx_z][skip])
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(f'Velocity Field at t = {t_real} s')
        plt.show()
        
        # Pressure Distribution
        plt.figure(figsize=(8,6))
        plt.contourf(X[:,:,idx_z]*L_star*1000, Y[:,:,idx_z]*W*1000, p_pred_dim[:,:,idx_z], levels=50, cmap='viridis')
        plt.colorbar(label='Dimensionless Pressure')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(f'Pressure Distribution at t = {t_real} s')
        plt.show()
        
        # Surfactant Concentration
        plt.figure(figsize=(8,6))
        plt.contourf(X[:,:,idx_z]*L_star*1000, Y[:,:,idx_z]*W*1000, C_pred_dim[:,:,idx_z], levels=50, cmap='plasma')
        plt.colorbar(label='Concentration (kg/kg)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(f'Surfactant Concentration at t = {t_real} s')
        plt.show()



def main():
    # Number of points
    N_col = 10000
    N_bnd = 10000
    N_init = 5000
    
    # Generate collocation points
    X_col = generate_collocation_points(N_col)
    X_col_tf = tf.convert_to_tensor(X_col, dtype=tf.float32)
    
    # Generate boundary points
    X_boundary = generate_boundary_points(N_bnd)
    
    # Generate initial points
    X_init = generate_initial_points(N_init)
    X_init_tf = tf.convert_to_tensor(X_init, dtype=tf.float32)
    
    # Initialize the model
    model = PINN()
    
    # Train the model
    loss_history, pde_loss_history, bc_loss_history, ic_loss_history = train_model(
        model, optimizer, X_col_tf, X_boundary, X_init_tf)
    
    # Evaluate and visualize
    times_to_evaluate = [1, 5, 10, 30, 60]
    evaluate_and_visualize(model, times_to_evaluate)
    
if __name__ == '__main__':
    main()




