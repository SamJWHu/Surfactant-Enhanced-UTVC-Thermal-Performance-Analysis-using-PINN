# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:26:34 2024

@author: SamJWHu
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING messages

# ---------------------------------------
# 1. Physical Constants and Properties
# ---------------------------------------
rho = 998         # Density of water at ~30°C (kg/m³)
mu = 1.0e-3       # Dynamic viscosity of water (Pa·s)
cp = 4182         # Specific heat capacity of water (J/(kg·K))
k = 0.6           # Thermal conductivity of water (W/(m·K))
D = 1.0e-9        # Diffusion coefficient of surfactant in water (m²/s)
sigma = 0.065     # Surface tension of the surfactant-water mixture (N/m)

# UTVC dimensions (m)
L = 0.2           # Length (200 mm)
W = 0.1           # Width (100 mm)
H = 0.0015        # Thickness (1.5 mm)

# Heat sources
Q_CPU = 30        # CPU heat (W)
Q_GPU = 50        # GPU heat (W)

# ---------------------------------------
# 2. Dimensionless Numbers
# ---------------------------------------
# Characteristic scales
L_star = L        # Length scale (m)
U_star = 0.01     # Velocity scale (m/s), estimated
T_star = 40       # Temperature difference scale (K), from 30°C to 70°C
C_star = 0.1      # Concentration scale (kg/kg), 0.1 (kg/kg)

# Reynolds number
Re = (rho * U_star * L_star) / mu
print(f"Reynolds number (Re): {Re:.2f}")

# Prandtl number
Pr = (mu * cp) / k
print(f"Prandtl number (Pr): {Pr:.2f}")

# Péclet numbers
Pe_T = Re * Pr
print(f"Péclet number for heat (Pe_T): {Pe_T:.2e}")

Pe_C = (U_star * L_star) / D
print(f"Péclet number for concentration (Pe_C): {Pe_C:.2e}")

# Nusselt number (for convection boundary condition)
Nu = 10.0  # Reduced value to represent realistic convection
print(f"Nusselt number (Nu): {Nu:.2f}")

# ---------------------------------------
# 3. Normalization and Denormalization Functions
# ---------------------------------------
def normalize_x(x):
    return x / L_star

def normalize_y(y):
    return y / W

def normalize_t(t, t_max=60):
    return t / t_max  # Normalize time with t_max = 60 seconds

def normalize_u(u):
    return u / U_star

def normalize_v(v):
    return v / U_star

def normalize_p(p):
    p_star = rho * U_star ** 2
    return p / p_star

def normalize_T(T):
    T_ambient = 30  # Ambient temperature in °C
    return (T - T_ambient) / T_star

def normalize_C(C):
    return C / C_star

def denormalize_T(T_dimless):
    T_ambient = 30  # Ambient temperature in °C
    return T_dimless * T_star + T_ambient

def denormalize_u(u_dimless):
    return u_dimless * U_star

def denormalize_v(v_dimless):
    return v_dimless * U_star

def denormalize_p(p_dimless):
    p_star = rho * U_star ** 2
    return p_dimless * p_star

def denormalize_C(C_dimless):
    return C_dimless * C_star

# ---------------------------------------
# 4. Neural Network Hyperparameters
# ---------------------------------------
input_dim = 3          # x, y, t
output_dim = 5         # u, v, p, T, C
num_hidden_layers = 6  # Reduced layers for faster training
num_neurons_per_layer = 40  # Reduced neurons per layer
activation_function = tf.nn.tanh

# Training parameters
learning_rate = 1e-3   # Reduced learning rate
epochs = 10000          # Adjusted number of epochs
batch_size = 1024      # Increased batch size for stability

# ---------------------------------------
# 5. Data Generation
# ---------------------------------------

# 5.1. Collocation Points
N_col = 20000  # Reduced number for faster training
x_col = np.random.uniform(0, L, (N_col, 1))
y_col = np.random.uniform(0, W, (N_col, 1))
t_col = np.random.uniform(0, 60, (N_col, 1))
X_col = np.hstack((normalize_x(x_col), normalize_y(y_col), normalize_t(t_col)))

# 5.2. Evaporation Regions (Heat Flux Boundary Conditions)
def is_in_cpu_region(x, y):
    return (x >= 0.3 * L) & (x <= 0.4 * L) & (y >= 0.45 * W) & (y <= 0.55 * W)

def is_in_gpu_region(x, y):
    return (x >= 0.6 * L) & (x <= 0.7 * L) & (y >= 0.45 * W) & (y <= 0.55 * W)

N_evap = 2000
x_evap = np.random.uniform(0, L, (N_evap, 1))
y_evap = np.random.uniform(0, W, (N_evap, 1))
t_evap = np.random.uniform(0, 60, (N_evap, 1))

cpu_mask = is_in_cpu_region(x_evap, y_evap).flatten()
gpu_mask = is_in_gpu_region(x_evap, y_evap).flatten()

X_evap_cpu = np.hstack((
    normalize_x(x_evap[cpu_mask]),
    normalize_y(y_evap[cpu_mask]),
    normalize_t(t_evap[cpu_mask])
))

X_evap_gpu = np.hstack((
    normalize_x(x_evap[gpu_mask]),
    normalize_y(y_evap[gpu_mask]),
    normalize_t(t_evap[gpu_mask])
))

# 5.3. Heat Fluxes in Dimensionless Form
q_cpu = Q_CPU / (k * T_star / L_star)
q_gpu = Q_GPU / (k * T_star / L_star)

# Normalize heat fluxes
q_cpu_norm = q_cpu / max(q_cpu, q_gpu)
q_gpu_norm = q_gpu / max(q_cpu, q_gpu)

# 5.4. Condensation Regions (Convection Boundary Conditions)
N_cond = 2000
x_cond_left = np.zeros((N_cond, 1))  # Left boundary (x=0)
x_cond_right = np.full((N_cond, 1), L)  # Right boundary (x=L)
y_cond = np.random.uniform(0, W, (N_cond, 1))
t_cond = np.random.uniform(0, 60, (N_cond, 1))

X_cond_left = np.hstack((
    normalize_x(x_cond_left),
    normalize_y(y_cond),
    normalize_t(t_cond)
))

X_cond_right = np.hstack((
    normalize_x(x_cond_right),
    normalize_y(y_cond),
    normalize_t(t_cond)
))

# 5.5. Adiabatic Boundaries (No Heat Transfer)
# Top boundary (y = W)
N_top = 1000
x_top = np.random.uniform(0, L, (N_top, 1))
y_top = np.full((N_top, 1), W)
t_top = np.random.uniform(0, 60, (N_top, 1))
X_top = np.hstack((
    normalize_x(x_top),
    normalize_y(y_top),
    normalize_t(t_top)
))

# Bottom boundary (y = 0)
N_bottom = 1000
x_bottom = np.random.uniform(0, L, (N_bottom, 1))
y_bottom = np.zeros((N_bottom, 1))
t_bottom = np.random.uniform(0, 60, (N_bottom, 1))
X_bottom = np.hstack((
    normalize_x(x_bottom),
    normalize_y(y_bottom),
    normalize_t(t_bottom)
))

# 5.6. Initial Conditions
N_init = 5000
x_init = np.random.uniform(0, L, (N_init, 1))
y_init = np.random.uniform(0, W, (N_init, 1))
t_init = np.zeros((N_init, 1))  # t = 0
X_init = np.hstack((
    normalize_x(x_init),
    normalize_y(y_init),
    normalize_t(t_init)
))

# Initial values (normalized)
u_init = np.zeros((N_init, 1))
v_init = np.zeros((N_init, 1))
T_init = np.zeros((N_init, 1))  # T' = 0 corresponds to ambient temperature
C_init = np.ones((N_init, 1))   # C' = 1 corresponds to initial concentration

# ---------------------------------------
# 6. Converting Data to TensorFlow Tensors
# ---------------------------------------
X_col_tf = tf.convert_to_tensor(X_col, dtype=tf.float32)
X_evap_cpu_tf = tf.convert_to_tensor(X_evap_cpu, dtype=tf.float32)
X_evap_gpu_tf = tf.convert_to_tensor(X_evap_gpu, dtype=tf.float32)
X_cond_left_tf = tf.convert_to_tensor(X_cond_left, dtype=tf.float32)
X_cond_right_tf = tf.convert_to_tensor(X_cond_right, dtype=tf.float32)
X_top_tf = tf.convert_to_tensor(X_top, dtype=tf.float32)
X_bottom_tf = tf.convert_to_tensor(X_bottom, dtype=tf.float32)
X_init_tf = tf.convert_to_tensor(X_init, dtype=tf.float32)

u_init_tf = tf.convert_to_tensor(u_init, dtype=tf.float32)
v_init_tf = tf.convert_to_tensor(v_init, dtype=tf.float32)
T_init_tf = tf.convert_to_tensor(T_init, dtype=tf.float32)
C_init_tf = tf.convert_to_tensor(C_init, dtype=tf.float32)

# ---------------------------------------
# 7. Model Definition
# ---------------------------------------
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
        for _ in range(num_hidden_layers):
            self.model.add(tf.keras.layers.Dense(
                num_neurons_per_layer, activation=activation_function,
                kernel_initializer='glorot_normal'))
        # Output layer with linear activation
        self.model.add(tf.keras.layers.Dense(output_dim, activation=None))
    
    def call(self, X):
        return self.model(X)

# Initialize the model
model = PINN()

# Optionally, display model summary
model.build((None, input_dim))
model.summary()

# ---------------------------------------
# 8. Loss Functions with Adjusted Weighting
# ---------------------------------------

# 8.1. PDE Residual Loss
def compute_pde_loss(model, X):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            output = model(X)
            u, v, p, T, C = tf.split(output, num_or_size_splits=5, axis=1)
        # First-order derivatives
        u_x = tape1.gradient(u, X)[:, 0:1]
        u_y = tape1.gradient(u, X)[:, 1:2]
        v_x = tape1.gradient(v, X)[:, 0:1]
        v_y = tape1.gradient(v, X)[:, 1:2]
        T_x = tape1.gradient(T, X)[:, 0:1]
        T_y = tape1.gradient(T, X)[:, 1:2]
        C_x = tape1.gradient(C, X)[:, 0:1]
        C_y = tape1.gradient(C, X)[:, 1:2]
        p_x = tape1.gradient(p, X)[:, 0:1]
        p_y = tape1.gradient(p, X)[:, 1:2]
        u_t = tape1.gradient(u, X)[:, 2:3]
        v_t = tape1.gradient(v, X)[:, 2:3]
        T_t = tape1.gradient(T, X)[:, 2:3]
        C_t = tape1.gradient(C, X)[:, 2:3]
    # Second-order derivatives
    u_xx = tape2.gradient(u_x, X)[:, 0:1]
    u_yy = tape2.gradient(u_y, X)[:, 1:2]
    v_xx = tape2.gradient(v_x, X)[:, 0:1]
    v_yy = tape2.gradient(v_y, X)[:, 1:2]
    T_xx = tape2.gradient(T_x, X)[:, 0:1]
    T_yy = tape2.gradient(T_y, X)[:, 1:2]
    C_xx = tape2.gradient(C_x, X)[:, 0:1]
    C_yy = tape2.gradient(C_y, X)[:, 1:2]
    
    # Delete tapes to free memory
    del tape1
    del tape2
    
    # Continuity equation residual
    continuity = u_x + v_y
    
    # Navier-Stokes equations residuals (dimensionless form)
    momentum_u = Re * (u_t + u * u_x + v * u_y) + p_x - (u_xx + u_yy)
    momentum_v = Re * (v_t + u * v_x + v * v_y) + p_y - (v_xx + v_yy)
    
    # Energy equation residual
    energy = Pe_T * (T_t + u * T_x + v * T_y) - (T_xx + T_yy)
    
    # Surfactant transport equation residual
    surfactant = Pe_C * (C_t + u * C_x + v * C_y) - (C_xx + C_yy)
    
    # Apply weighting factors to balance the loss terms
    # Since Pe_T and Pe_C are large, we scale the energy and surfactant residuals
    scaling_factor_energy = 1.0 / Pe_T
    scaling_factor_surfactant = 1.0 / Pe_C
    
    # Mean Squared Errors
    mse_continuity = tf.reduce_mean(tf.square(continuity))
    mse_momentum_u = tf.reduce_mean(tf.square(momentum_u))
    mse_momentum_v = tf.reduce_mean(tf.square(momentum_v))
    mse_energy = tf.reduce_mean(tf.square(energy * scaling_factor_energy))
    mse_surfactant = tf.reduce_mean(tf.square(surfactant * scaling_factor_surfactant))
    
    total_pde_loss = mse_continuity + mse_momentum_u + mse_momentum_v + mse_energy + mse_surfactant
    
    return total_pde_loss

# 8.2. Boundary Conditions Loss
def compute_boundary_loss(model, X_evap_cpu, X_evap_gpu, X_cond_left, X_cond_right, X_top, X_bottom):
    # Evaporation regions (Heat Flux Boundary Conditions)
    # CPU Region
    with tf.GradientTape() as tape:
        tape.watch(X_evap_cpu)
        output_cpu = model(X_evap_cpu)
        T_cpu = output_cpu[:, 3:4]  # Temperature component
    T_n_cpu = tape.gradient(T_cpu, X_evap_cpu)[:, 1:2]  # ∂T/∂y (assuming normal in y-direction)
    q_cpu_pred = -T_n_cpu  # Negative gradient represents heat flux
    mse_q_cpu = tf.reduce_mean(tf.square(q_cpu_pred - q_cpu_norm))
    
    # GPU Region
    with tf.GradientTape() as tape:
        tape.watch(X_evap_gpu)
        output_gpu = model(X_evap_gpu)
        T_gpu = output_gpu[:, 3:4]
    T_n_gpu = tape.gradient(T_gpu, X_evap_gpu)[:, 1:2]
    q_gpu_pred = -T_n_gpu
    mse_q_gpu = tf.reduce_mean(tf.square(q_gpu_pred - q_gpu_norm))
    
    # Condensation Regions (Convection Boundary Conditions)
    # Left Boundary
    with tf.GradientTape() as tape:
        tape.watch(X_cond_left)
        output_cond_left = model(X_cond_left)
        T_cond_left = output_cond_left[:, 3:4]
    T_n_cond_left = tape.gradient(T_cond_left, X_cond_left)[:, 0:1]  # ∂T/∂x
    mse_cond_left = tf.reduce_mean(tf.square(-T_n_cond_left + Nu * T_cond_left))
    
    # Right Boundary
    with tf.GradientTape() as tape:
        tape.watch(X_cond_right)
        output_cond_right = model(X_cond_right)
        T_cond_right = output_cond_right[:, 3:4]
    T_n_cond_right = tape.gradient(T_cond_right, X_cond_right)[:, 0:1]  # ∂T/∂x
    mse_cond_right = tf.reduce_mean(tf.square(T_n_cond_right + Nu * T_cond_right))
    
    # Adiabatic Boundaries (No Heat Transfer)
    # Top Boundary
    with tf.GradientTape() as tape:
        tape.watch(X_top)
        output_top = model(X_top)
        T_top = output_top[:, 3:4]
    T_n_top = tape.gradient(T_top, X_top)[:, 1:2]  # ∂T/∂y
    mse_Tn_top = tf.reduce_mean(tf.square(T_n_top))  # Should be 0
    
    # Bottom Boundary
    with tf.GradientTape() as tape:
        tape.watch(X_bottom)
        output_bottom = model(X_bottom)
        T_bottom = output_bottom[:, 3:4]
    T_n_bottom = tape.gradient(T_bottom, X_bottom)[:, 1:2]
    mse_Tn_bottom = tf.reduce_mean(tf.square(T_n_bottom))  # Should be 0
    
    # Total Boundary Condition Loss
    total_bc_loss = (mse_q_cpu + mse_q_gpu +
                     mse_cond_left + mse_cond_right +
                     mse_Tn_top + mse_Tn_bottom)
    
    return total_bc_loss

# 8.3. Initial Conditions Loss with Reduced Weight
def compute_initial_loss(model, X_init, u_init, v_init, T_init, C_init):
    output = model(X_init)
    u_pred, v_pred, _, T_pred, C_pred = tf.split(output, num_or_size_splits=5, axis=1)
    mse_u = tf.reduce_mean(tf.square(u_pred - u_init))
    mse_v = tf.reduce_mean(tf.square(v_pred - v_init))
    mse_T = tf.reduce_mean(tf.square(T_pred - T_init))
    mse_C = tf.reduce_mean(tf.square(C_pred - C_init))
    total_ic_loss = mse_u + mse_v + mse_T + mse_C
    return total_ic_loss

# 8.4. Combined Loss Function with Weighting Factors
def loss_function(model, X_col, X_evap_cpu, X_evap_gpu, X_cond_left, X_cond_right,
                  X_top, X_bottom, X_init, u_init, v_init, T_init, C_init):
    loss_pde = compute_pde_loss(model, X_col)
    loss_bc = compute_boundary_loss(model, X_evap_cpu, X_evap_gpu, X_cond_left,
                                    X_cond_right, X_top, X_bottom)
    loss_ic = compute_initial_loss(model, X_init, u_init, v_init, T_init, C_init)
    # Apply weighting factors
    total_loss = loss_pde + 10.0 * loss_bc + 0.1 * loss_ic
    return total_loss, loss_pde, loss_bc, loss_ic

# ---------------------------------------
# 9. Optimizer Setup with Learning Rate Scheduler
# ---------------------------------------
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

# ---------------------------------------
# 10. Training Function with Gradient Clipping
# ---------------------------------------
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        total_loss, loss_pde, loss_bc, loss_ic = loss_function(
            model, X_col_tf, X_evap_cpu_tf, X_evap_gpu_tf, X_cond_left_tf,
            X_cond_right_tf, X_top_tf, X_bottom_tf, X_init_tf,
            u_init_tf, v_init_tf, T_init_tf, C_init_tf)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    # Gradient clipping
    gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, loss_pde, loss_bc, loss_ic

# ---------------------------------------
# 11. Training Loop
# ---------------------------------------
# Initialize the model
model = PINN()

# Lists to store losses for plotting
total_loss_history = []
pde_loss_history = []
bc_loss_history = []
ic_loss_history = []

start_time = time.time()

for epoch in range(epochs):
    total_loss_value, loss_pde_value, loss_bc_value, loss_ic_value = train_step()
    total_loss_history.append(total_loss_value.numpy())
    pde_loss_history.append(loss_pde_value.numpy())
    bc_loss_history.append(loss_bc_value.numpy())
    ic_loss_history.append(loss_ic_value.numpy())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss_value.numpy():.4e}, "
              f"PDE Loss: {loss_pde_value.numpy():.4e}, "
              f"BC Loss: {loss_bc_value.numpy():.4e}, "
              f"IC Loss: {loss_ic_value.numpy():.4e}")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# ---------------------------------------
# 12. Evaluation and Visualization
# ---------------------------------------

# 12.1. Visualization of Training Loss
plt.figure(figsize=(10, 6))
plt.semilogy(total_loss_history, label='Total Loss')
plt.semilogy(pde_loss_history, label='PDE Loss')
plt.semilogy(bc_loss_history, label='Boundary Loss')
plt.semilogy(ic_loss_history, label='Initial Condition Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss History')
plt.grid(True)
plt.show()

# 12.2. Generate Test Points for Evaluation at Multiple Times
def generate_test_points(t_physical):
    """
    Generate test points at a specific physical time.
    """
    x_test = np.linspace(0, L, N_test)
    y_test = np.linspace(0, W, N_test)
    t_test = np.full((N_test, N_test), t_physical)  # Physical time
    t_test_normalized = normalize_t(t_test)
    
    X_grid, Y_grid = np.meshgrid(x_test, y_test)
    X_test = np.hstack([
        normalize_x(X_grid.flatten()[:, None]),
        normalize_y(Y_grid.flatten()[:, None]),
        t_test_normalized.flatten()[:, None]
    ])
    X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    return X_test_tf, X_grid, Y_grid

# Define test times in seconds
test_times = [1, 5, 10, 30, 60]

# Define number of test points
N_test = 100

# 12.3. Define Plotting Function
def plot_distributions(model, t_physical):
    """
    Plot Temperature, Velocity, Pressure, and Concentration distributions at a given time.
    """
    X_test_tf, X_grid, Y_grid = generate_test_points(t_physical)
    
    # Model prediction
    output_pred = model(X_test_tf)
    u_pred, v_pred, p_pred, T_pred, C_pred = tf.split(output_pred, num_or_size_splits=5, axis=1)
    
    # Reshape for plotting
    u_pred = u_pred.numpy().reshape(N_test, N_test)
    v_pred = v_pred.numpy().reshape(N_test, N_test)
    p_pred = p_pred.numpy().reshape(N_test, N_test)
    T_pred = T_pred.numpy().reshape(N_test, N_test)
    C_pred = C_pred.numpy().reshape(N_test, N_test)
    
    # Denormalize temperature and velocities
    T_pred_dim = denormalize_T(T_pred)
    u_pred_dim = denormalize_u(u_pred)
    v_pred_dim = denormalize_v(v_pred)
    p_pred_dim = denormalize_p(p_pred)
    C_pred_dim = denormalize_C(C_pred)
    
    # Create meshgrid for plotting (in mm)
    X_dim = X_grid * 1000  # Convert to mm
    Y_dim = Y_grid * 1000  # Convert to mm
    
    # Plot Temperature Distribution
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_dim, Y_dim, T_pred_dim, levels=50, cmap='jet')
    plt.colorbar(cp, label='Temperature (°C)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Temperature Distribution at t = {t_physical} s')
    plt.show()
    
    # Plot Velocity Field
    plt.figure(figsize=(8, 6))
    plt.quiver(X_dim, Y_dim, u_pred_dim, v_pred_dim, scale=50)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Velocity Field at t = {t_physical} s')
    plt.show()
    
    # Plot Pressure Distribution
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_dim, Y_dim, p_pred_dim, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Pressure (Pa)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Pressure Distribution at t = {t_physical} s')
    plt.show()
    
    # Plot Surfactant Concentration
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_dim, Y_dim, C_pred_dim, levels=50, cmap='plasma')
    plt.colorbar(cp, label='Concentration (kg/kg)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Surfactant Concentration at t = {t_physical} s')
    plt.show()

# 12.4. Plot Distributions at Specified Times
for t_physical in test_times:
    plot_distributions(model, t_physical)





