#!/usr/bin/env python


# Import standard Python modules.
import datetime
# import importlib
import os
import platform
import sys

# Import 3rd-party modules.
import numpy as np

# Import TensorFlow.
import tensorflow as tf

# Import project modules.
from nnde.math.trainingdata import create_training_grid2


# Use 64-bit math in TensorFlow.
tf.keras.backend.set_floatx('float64')


# Inlet conditions
rho_0 = 1.0
vx_0  = 0.0
vy_0  = 0.0
vz_0  = 0.0
# Bx_0  = 0.0
By_0  = 1.0
Bz_0  = 0.0
p_0   = 1.0

# Outlet conditions
rho_1 = 0.125
vx_1  = 0.0
vy_1  = 0.0
vz_1  = 0.0
# Bx_1  = 0.0
By_1  = -1.0
Bz_1  = 0.0
p_1   = 0.1

# Constants
# gamma = 2.0


def print_system_information():
    print("System report:")
    print(datetime.datetime.now())
    print("Host name: %s" % platform.node())
    print("OS: %s" % platform.platform())
    print("uname:", platform.uname())
    print("Python version: %s" % sys.version)
    print("Python build:", platform.python_build())
    print("Python compiler: %s" % platform.python_compiler())
    print("Python implementation: %s" % platform.python_implementation())
    print("Python file: %s" % __file__)


def create_output_directory(path=None):
    path_noext, ext = os.path.splitext(path)
    output_dir = path_noext
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir



def create_training_data(*n_train):
    x_train = np.array(create_training_grid2(*n_train))
    return x_train


def build_model(H, w0_range, u0_range, v0_range):
    hidden_layer = tf.keras.layers.Dense(
        units=H, use_bias=True,
        activation=tf.keras.activations.sigmoid,
        kernel_initializer=tf.keras.initializers.RandomUniform(*w0_range),
        bias_initializer=tf.keras.initializers.RandomUniform(*u0_range)
    )
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(*v0_range),
        use_bias=False,
    )
    model = tf.keras.Sequential([hidden_layer, output_layer])
    return model


# Define the differential equations using TensorFlow operations.

# xt is the vector (x, t)
# y is the list of vectors (rho, vx, vy, vz, By, Bz, E)
# dely is the list of gradients (del_rho, del_vx, del_vy, del_vz, delBy, del_Bz, del_p)

# @tf.function
# def pde_rho(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     G = drho_dt + rho*dvx_dx + drho_dx*vx
#     return G

# @tf.function
# def pde_vx(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     dpstar_dx = dp_dx + dBx_dx + dBy_dx + dBz_dx
#     G = rho*dvx_dt + drho_dt*vx + rho*2*vx*dvx_dx + drho_dx*vx**2 + dpstar_dx - 2*Bx*dBx_dx
#     return G

# @tf.function
# def pde_vy(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     G = rho*dvy_dt + drho_dt*vy + rho*vx*dvy_dx + rho*dvx_dx*vy + drho_dx*vx*vy - Bx*dBy_dx - dBx_dx*By
#     return G

# @tf.function
# def pde_vz(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     G = rho*dvz_dt + drho_dt*vz + rho*vx*dvz_dx + rho*dvx_dx*vz + drho_dx*vx*vz - Bx*dBz_dx - dBx_dx*Bz
#     return G

# @tf.function
# def pde_By(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     G = dBy_dt + By*dvx_dx + dBy_dx*vx - Bx*dvy_dx - dBx_dx*vy
#     return G

# @tf.function
# def pde_Bz(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     G = dBz_dt + Bz*dvx_dx + dBz_dx*vx - Bx*dvz_dx - dBx_dx*vz
#     return G

# @tf.function
# def pde_p(xt, y, dely):
#     x   = xt[:, 0]
#     t   = xt[:, 1]
#     rho = y[0][:, 0]
#     vx  = y[1][:, 0]
#     vy  = y[2][:, 0]
#     vz  = y[3][:, 0]
#     Bx  = 0
#     By  = y[4][:, 0]
#     Bz  = y[5][:, 0]
#     p   = y[6][:, 0]
#     drho_dx = dely[0][:, 0, 0]
#     drho_dt = dely[0][:, 0, 1]
#     dvx_dx  = dely[1][:, 0, 0]
#     dvx_dt  = dely[1][:, 0, 1]
#     dvy_dx  = dely[2][:, 0, 0]
#     dvy_dt  = dely[2][:, 0, 1]
#     dvz_dx  = dely[3][:, 0, 0]
#     dvz_dt  = dely[3][:, 0, 1]
#     dBx_dx  = 0
#     dBx_dt  = 0
#     dBy_dx  = dely[4][:, 0, 0]
#     dBy_dt  = dely[4][:, 0, 1]
#     dBz_dx  = dely[5][:, 0, 0]
#     dBz_dt  = dely[5][:, 0, 1]
#     dp_dx   = dely[6][:, 0, 0]
#     dp_dt   = dely[6][:, 0, 1]
#     pstar = p + 0.5*(Bx**2 + By**2 + Bz**2)
#     dpstar_dx = dp_dx + dBx_dx + dBy_dx + dBz_dx
#     E = 0.5*rho*(vx**2 + vy**2 + vz**2) + p/(gamma - 1.0) + 0.5*(Bx**2 + By**2 + Bz**2)
#     dE_dx = (
#         rho*(vx*dvx_dx + vy*dvy_dx + vz*dvz_dx) + dp_dx/(gamma - 1)
#         + Bx*dBx_dx + By*dBy_dx + Bz*dBz_dx
#     )
#     dE_dt = (
#         rho*(vx*dvx_dt + vy*dvy_dt + vz*dvz_dt) + dp_dt/(gamma - 1)
#         + Bx*dBx_dt + By*dBy_dt + Bz*dBz_dt
#     )
#     G = (
#         dE_dt + (E + pstar)*dvx_dx + (dE_dx + dpstar_dx)*vx
#         - Bx*(Bx*dvx_dx + dBx_dx*vx + By*dvy_dx + dBy_dx*vy + Bz*dvz_dx + dBz_dx*vz)
#         - dBx_dx*(Bx*vx + By*vy + Bz*vz)
#     )
#     return G


# Define the boundary conditions.

def f0_rho(xt):
    return rho_0

def f1_rho(xt):
    return rho_1

def g0_rho(xt):
    x = xt[:, 0]
    t = xt[:, 1]
    g0 = np.zeros_like(x)
    g0[np.where(x <= 0.5)] = rho_0
    g0[np.where(x > 0.5)] = rho_1
    g0 = tf.Variable(g0)
    return g0

# def f0_vx(xt):
#     return vx_0

# def f1_vx(xt):
#     return vx_1

# def g0_vx(xt):
#     (x, t) = xt
#     if x <= 0.5:
#         g0 = vx_0
#     else:
#         g0 = vx_1
#     return g0

# def f0_vy(xt):
#     return vy_0

# def f1_vy(xt):
#     return vy_1

# def g0_vy(xt):
#     (x, t) = xt
#     if x <= 0.5:
#         g0 = vy_0
#     else:
#         g0 = vy_1
#     return g0

# def f0_vz(xt):
#     return vz_0

# def f1_vz(xt):
#     return vz_1

# def g0_vz(xt):
#     (x, t) = xt
#     if x <= 0.5:
#         g0 = vz_0
#     else:
#         g0 = vz_1
#     return g0

# def f0_By(xt):
#     return By_0

# def f1_By(xt):
#     return By_1

# def g0_By(xt):
#     (x, t) = xt
#     if x <= 0.5:
#         g0 = By_0
#     else:
#         g0 = By_1
#     return g0

# def f0_Bz(xt):
#     return Bz_0

# def f1_Bz(xt):
#     return Bz_1

# def g0_Bz(xt):
#     (x, t) = xt
#     if x <= 0.5:
#         g0 = Bz_0
#     else:
#         g0 = Bz_1
#     return g0

# def f0_p(xt):
#     return p_0

# def f1_p(xt):
#     return p_1

# def g0_p(xt):
#     (x, t) = xt
#     if x <= 0.5:
#         g0 = p_0
#     else:
#         g0 = p_1
#     return g0


# Define the trial functions.

x0t = None
x1t = None

@tf.function
def Ytrial_rho(xt, N):
    x = xt[:, 0]
    t = xt[:, 1]
    A = (
        (1 - x)*f0_rho(xt) + x*f1_rho(xt)
        + (1 - t)*(g0_rho(xt) - ((1 - x)*g0_rho(x0t) + x*g0_rho(x1t)))
    )
    P = x*(1 - x)*t
    Y = A + P*N[:, 0]
    return Y

# @tf.function
# def Ytrial_vx(xt, N):
#     x = xt[:, 0]
#     t = xt[:, 1]
#     x0t_np = xt.numpy()
#     x0t_np[:, 0] = 0
#     x0t = tf.Variable(x0t_np)
#     x1t_np = xt.numpy()
#     x1t_np[:, 0] = 1
#     x1t = tf.Variable(x1t_np)
#     A = (
#         (1 - x)*f0_vx(xt) + x*f1_vx(xt)
#         + (1 - t)*(g0_vx(xt) - ((1 - x)*g0_vx(x0t) + x*g0_vx(x1t)))
#     )
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# @tf.function
# def Ytrial_vy(xt, N):
#     x = xt[:, 0]
#     t = xt[:, 1]
#     x0t_np = xt.numpy()
#     x0t_np[:, 0] = 0
#     x0t = tf.Variable(x0t_np)
#     x1t_np = xt.numpy()
#     x1t_np[:, 0] = 1
#     x1t = tf.Variable(x1t_np)
#     A = (
#         (1 - x)*f0_vy(xt) + x*f1_vy(xt)
#         + (1 - t)*(g0_vy(xt) - ((1 - x)*g0_vy(x0t) + x*g0_vy(x1t)))
#     )
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# @tf.function
# def Ytrial_vz(xt, N):
#     x = xt[:, 0]
#     t = xt[:, 1]
#     x0t_np = xt.numpy()
#     x0t_np[:, 0] = 0
#     x0t = tf.Variable(x0t_np)
#     x1t_np = xt.numpy()
#     x1t_np[:, 0] = 1
#     x1t = tf.Variable(x1t_np)
#     A = (
#         (1 - x)*f0_vz(xt) + x*f1_vz(xt)
#         + (1 - t)*(g0_vz(xt) - ((1 - x)*g0_vz(x0t) + x*g0_vz(x1t)))
#     )
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# @tf.function
# def Ytrial_By(xt, N):
#     x = xt[:, 0]
#     t = xt[:, 1]
#     x0t_np = xt.numpy()
#     x0t_np[:, 0] = 0
#     x0t = tf.Variable(x0t_np)
#     x1t_np = xt.numpy()
#     x1t_np[:, 0] = 1
#     x1t = tf.Variable(x1t_np)
#     A = (
#         (1 - x)*f0_By(xt) + x*f1_By(xt)
#         + (1 - t)*(g0_By(xt) - ((1 - x)*g0_By(x0t) + x*g0_By(x1t)))
#     )
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# @tf.function
# def Ytrial_Bz(xt, N):
#     x = xt[:, 0]
#     t = xt[:, 1]
#     x0t_np = xt.numpy()
#     x0t_np[:, 0] = 0
#     x0t = tf.Variable(x0t_np)
#     x1t_np = xt.numpy()
#     x1t_np[:, 0] = 1
#     x1t = tf.Variable(x1t_np)
#     A = (
#         (1 - x)*f0_Bz(xt) + x*f1_Bz(xt)
#         + (1 - t)*(g0_Bz(xt) - ((1 - x)*g0_Bz(x0t) + x*g0_Bz(x1t)))
#     )
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y

# @tf.function
# def Ytrial_p(xt, N):
#     x = xt[:, 0]
#     t = xt[:, 1]
#     x0t_np = xt.numpy()
#     x0t_np[:, 0] = 0
#     x0t = tf.Variable(x0t_np)
#     x1t_np = xt.numpy()
#     x1t_np[:, 0] = 1
#     x1t = tf.Variable(x1t_np)
#     A = (
#         (1 - x)*f0_p(xt) + x*f1_p(xt)
#         + (1 - t)*(g0_p(xt) - ((1 - x)*g0_p(x0t) + x*g0_p(x1t)))
#     )
#     P = x*(1 - x)*t
#     Y = A + P*N[:, 0]
#     return Y


if __name__ == "__main__":
    """Begin main program."""

    print_system_information()

    # Set up the output directory.
    eq_name = "1dmhd00"
    path = os.path.join(".", eq_name)
    output_dir = create_output_directory(path)

    # Define the hyperparameters.

    # Training optimizer
    optimizer_name = "Adam"

    # Initial parameter ranges
    w0_range = [-0.1, 0.1]
    u0_range = [-0.1, 0.1]
    v0_range = [-0.1, 0.1]

    # Maximum number of training epochs.
    max_epochs = 1000

    # Learning rate.
    learning_rate = 0.01

    # Absolute tolerance for consecutive loss function values to indicate convergence.
    # tol = 1e-6

    # Number of hidden nodes.
    H = 10

    # Number of dimensions
    m = 2

    # Number of training points in each dimension.
    nx_train = 11
    nt_train = 11
    n_train = nx_train*nt_train

    # Number of validation points in each dimension.
    # nx_val = 21
    # nt_val = 21
    # n_val = nx_val*nt_val

    # Random number generator seed.
    random_seed = 0

    # Create and save the training data.
    xt_train = create_training_data(nx_train, nt_train)
    x_train = xt_train[::nt_train, 0]
    t_train = xt_train[:nt_train, 1]
    np.savetxt(os.path.join(output_dir,'xt_train.dat'), xt_train)

    # Create copies of the training data with all x = 0 and all x = 1.
    x0t_train = create_training_data(nx_train, nt_train)
    x0t_train[:, 0] = 0
    np.savetxt(os.path.join(output_dir,'x0t_train.dat'), x0t_train)
    x1t_train = create_training_data(nx_train, nt_train)
    x1t_train[:, 0] = 1
    np.savetxt(os.path.join(output_dir,'x1t_train.dat'), x1t_train)
    x0t = x0t_train
    x1t = x1t_train

    # Build the models.
    model_rho = build_model(H, w0_range, u0_range, v0_range)
    # model_vx  = build_model(H, w0_range, u0_range, v0_range)
    # model_vy  = build_model(H, w0_range, u0_range, v0_range)
    # model_vz  = build_model(H, w0_range, u0_range, v0_range)
    # model_By  = build_model(H, w0_range, u0_range, v0_range)
    # model_Bz  = build_model(H, w0_range, u0_range, v0_range)
    # model_p   = build_model(H, w0_range, u0_range, v0_range)

    # Create the optimizers.
    optimizer_rho = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer_vx  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer_vy  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer_vz  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer_By  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer_Bz  = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer_p   = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the models.

    # Create history variables.
    losses_rho = []
    # losses_vx  = []
    # losses_vy  = []
    # losses_vz  = []
    # losses_By  = []
    # losses_Bz  = []
    # losses_p   = []
    losses     = []

    phist_rho = []
    # phist_vx  = []
    # phist_vy  = []
    # phist_vz  = []
    # phist_By  = []
    # phist_Bz  = []
    # phist_p   = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(random_seed)

    # Rename the training data Variable for convenience.
    # shape (n_train, m)
    xt_train_var = tf.Variable(xt_train, name="xt_train")
    xt = xt_train_var
    # x = xt[:, 0]
    # t = xt[:, 1]


    # Clear the convergence flag to start.
    converged = False

    print("Hyperparameters: n_train = %s, H = %s, max_epochs = %s, optimizer = %s, learning_rate = %s"
          % (n_train, H, max_epochs, optimizer_name, learning_rate))
    t_start = datetime.datetime.now()
    print("Training started at", t_start)

    # <HACK>
    L = tf.Variable(0)
    # </HACK>

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the training points.
                N_rho = model_rho(xt)
                # N_vx  = model_vx(xt)
                # N_vy  = model_vy(xt)
                # N_vz  = model_vz(xt)
                # N_By  = model_By(xt)
                # N_Bz  = model_Bz(xt)
                # N_p   = model_p(xt)

                # Compute the trial solutions.
                # rho = Ytrial_rho(xt, N_rho)
    # #             vx  = Ytrial_vx(xt, N_vx)
    # #             vy  = Ytrial_vy(xt, N_vy)
    # #             vz  = Ytrial_vz(xt, N_vz)
    # #             Bx  = 0
    # #             By  = Ytrial_By(xt, N_By)
    # #             Bz  = Ytrial_Bz(xt, N_Bz)
    # #             p   =  Ytrial_p(xt, p)

    # #         # Compute the gradients of the trial solutions wrt inputs.
    # #         del_rho = tape0.gradient(rho, xt)
    # #         del_vx  = tape0.gradient(vx, xt)
    # #         del_vy  = tape0.gradient(vy, xt)
    # #         del_vz  = tape0.gradient(vz, xt)
    # #         del_Bx  = 0
    # #         del_By  = tape0.gradient(By, xt)
    # #         del_Bz  = tape0.gradient(Bz, xt)
    # #         del_p   = tape0.gradient(p, xt)

    # #         # Compute the estimates of the differential equations.
    # #         y = [rho, vx, vy, vz, Bz, By, Bz, p]
    # #         del_y = [del_rho, del_vx, del_vy, del_vz, del_Bx, del_By, del_Bz, del_p]
    # #         G_rho = pde_rho(xt, y, del_y)
    # #         G_vx  =  pde_vx(xt, y, del_y)
    # #         G_vy  =  pde_vy(xt, y, del_y)
    # #         G_vz  =  pde_vz(xt, y, del_y)
    # #         G_By  =  pde_By(xt, y, del_y)
    # #         G_Bz  =  pde_Bz(xt, y, del_y)
    # #         G_p   =   pde_p(xt, y, del_y)

    # #         # Compute the loss functions.
    # #         L_rho = tf.math.sqrt(tf.reduce_sum(G_rho**2)/n_train)
    # #         L_vx  = tf.math.sqrt(tf.reduce_sum(G_vx**2) /n_train)
    # #         L_vy  = tf.math.sqrt(tf.reduce_sum(G_vy**2) /n_train)
    # #         L_vz  = tf.math.sqrt(tf.reduce_sum(G_vz**2) /n_train)
    # #         L_By  = tf.math.sqrt(tf.reduce_sum(G_By**2) /n_train)
    # #         L_Bz  = tf.math.sqrt(tf.reduce_sum(G_Bz**2) /n_train)
    # #         L_p   = tf.math.sqrt(tf.reduce_sum(G_p**2)  /n_train)
    # #         L = L_rho + L_vx + L_vy + L_vz + L_By + L_Bz + L_p

    # #     # Save the current losses.
    # #     losses_rho.append(L_rho.numpy())
    # #     losses_vx.append( L_vx.numpy())
    # #     losses_vy.append( L_vy.numpy())
    # #     losses_vz.append( L_vz.numpy())
    # #     losses_By.append( L_By.numpy())
    # #     losses_Bz.append( L_Bz.numpy())
    # #     losses_p.append(  L_p.numpy())
    # #     losses.append(    L.numpy())

    # #     # Check for convergence.
    # #     if epoch > 1:
    # #         loss_delta = losses[-1] - losses[-2]
    # #         if abs(loss_delta) <= tol:
    # #             converged = True
    # #             break

    # #     # Compute the gradient of the loss function wrt the network parameters.
    # #     pgrad_rho = tape1.gradient(L, model_rho.trainable_variables)
    # #     pgrad_vx  = tape1.gradient(L, model_vx.trainable_variables)
    # #     pgrad_vy  = tape1.gradient(L, model_vy.trainable_variables)
    # #     pgrad_vz  = tape1.gradient(L, model_vz.trainable_variables)
    # #     pgrad_By  = tape1.gradient(L, model_By.trainable_variables)
    # #     pgrad_Bz  = tape1.gradient(L, model_Bz.trainable_variables)
    # #     pgrad_p   = tape1.gradient(L, model_p.trainable_variables)

    # #     # Save the parameters used in this epoch.
    # #     phist_rho.append(
    # #         np.hstack(
    # #             (model_rho.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_rho.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_rho.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )
    # #     phist_vx.append(
    # #         np.hstack(
    # #             (model_vx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_vx.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_vx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )
    # #     phist_vy.append(
    # #         np.hstack(
    # #             (model_vy.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_vy.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_vy.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )
    # #     phist_vz.append(
    # #         np.hstack(
    # #             (model_vz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_vz.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_vz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )
    # #     phist_By.append(
    # #         np.hstack(
    # #             (model_By.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_By.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_By.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )
    # #     phist_Bz.append(
    # #         np.hstack(
    # #             (model_Bz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_Bz.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_Bz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )
    # #     phist_p.append(
    # #         np.hstack(
    # #             (model_p.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #              model_p.trainable_variables[1].numpy(),       # u (H,) row vector
    # #              model_p.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #         )
    # #     )

    # #     # Update the parameters for this epoch.
    # #     optimizer_rho.apply_gradients(zip(pgrad_rho, model_rho.trainable_variables))
    # #     optimizer_vx.apply_gradients(zip( pgrad_vx,  model_vx.trainable_variables))
    # #     optimizer_vy.apply_gradients(zip( pgrad_vy,  model_vy.trainable_variables))
    # #     optimizer_vz.apply_gradients(zip( pgrad_vz,  model_vz.trainable_variables))
    # #     optimizer_By.apply_gradients(zip( pgrad_By,  model_By.trainable_variables))
    # #     optimizer_Bz.apply_gradients(zip( pgrad_Bz,  model_Bz.trainable_variables))
    # #     optimizer_p.apply_gradients( zip( pgrad_p,   model_p.trainable_variables))

        if epoch % 100 == 0:
            print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))

    # Save the parameters used in the last epoch.
    phist_rho.append(
        np.hstack(
            (model_rho.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
             model_rho.trainable_variables[1].numpy(),       # u (H,) row vector
             model_rho.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )
    # # phist_vx.append(
    # #     np.hstack(
    # #         (model_vx.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #          model_vx.trainable_variables[1].numpy(),       # u (H,) row vector
    # #          model_vx.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #     )
    # # )
    # # phist_vy.append(
    # #     np.hstack(
    # #         (model_vy.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #          model_vy.trainable_variables[1].numpy(),       # u (H,) row vector
    # #          model_vy.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #     )
    # # )
    # # phist_vz.append(
    # #     np.hstack(
    # #         (model_vz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #          model_vz.trainable_variables[1].numpy(),       # u (H,) row vector
    # #          model_vz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #     )
    # # )
    # # phist_By.append(
    # #     np.hstack(
    # #         (model_By.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #          model_By.trainable_variables[1].numpy(),       # u (H,) row vector
    # #          model_By.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #     )
    # # )
    # # phist_Bz.append(
    # #     np.hstack(
    # #         (model_Bz.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #          model_Bz.trainable_variables[1].numpy(),       # u (H,) row vector
    # #          model_Bz.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #     )
    # # )
    # # phist_p.append(
    # #     np.hstack(
    # #         (model_p.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
    # #          model_p.trainable_variables[1].numpy(),       # u (H,) row vector
    # #          model_p.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    # #     )
    # # )

    # Count the last epoch.
    n_epochs = epoch + 1

    t_stop = datetime.datetime.now()
    print("Training stopped at", t_stop)
    t_elapsed = t_stop - t_start
    print("Total training time was %s seconds." % t_elapsed.total_seconds())
    print("Epochs: %d" % n_epochs)
    print("Final value of loss function: %f" % losses[-1])
    print("converged = %s" % converged)


    # # Save the parameter histories.
    # np.savetxt(os.path.join(output_dir, 'phist_rho.dat'), np.array(phist_rho))
    # np.savetxt(os.path.join(output_dir, 'phist_vx.dat'),  np.array(phist_vx))
    # np.savetxt(os.path.join(output_dir, 'phist_vy.dat'),  np.array(phist_vy))
    # np.savetxt(os.path.join(output_dir, 'phist_vz.dat'),  np.array(phist_vz))
    # np.savetxt(os.path.join(output_dir, 'phist_By.dat'),  np.array(phist_By))
    # np.savetxt(os.path.join(output_dir, 'phist_Bz.dat'),  np.array(phist_Bz))
    # np.savetxt(os.path.join(output_dir, 'phist_p.dat'),  np.array(phist_p))
