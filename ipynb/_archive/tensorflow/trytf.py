# Import standard Python modules.
import datetime
#import importlib
from itertools import repeat
from math import exp
import os
import platform
import sys

# Import 3rd-party modules.
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

# Import TensorFlow.
import tensorflow as tf

class Equation:
    name = "lagaris05"

    @staticmethod
    def G(xy, Y, delY, del2Y):
        """Differential equation in standard form."""
        (x, y) = xy
        (d2Y_dx2, d2Y_dy2) = del2Y
        _G = d2Y_dx2 + d2Y_dy2 - exp(-x)*(x - 2 + y**3 + 6*y)
        return _G

    @staticmethod
    def f0(xy):
        """Boundary condition at (x, y) = (0, y)."""
        (x, y) = xy
        return y**3

    @staticmethod
    def f1(xy):
        """Boundary condition at (x, y) = (1, y)."""
        (x, y) = xy
        return (1 + y**3)*exp(-1)

    @staticmethod
    def g0(xy):
        """Boundary condition at (x, y) = (x, 0)."""
        (x, y) = xy
        return x*exp(-x)

    @staticmethod
    def g1(xy):
        """Boundary condition at (x, y) = (x, 1)."""
        (x, y) = xy
        return exp(-x)*(x + 1)

    @staticmethod
    def Ya(xy):
        """Analytical solution."""
        (x, y) = xy
        _Ya = exp(-x)*(x + y**3)
        return _Ya

    @staticmethod
    def dYa_dx(xy):
        """Analytical dY/dx."""
        (x, y) = xy
        _dYa_dx = exp(-x)*(1 - x - y**3)
        return _dYa_dx

    @staticmethod
    def dYa_dy(xy):
        """Analytical dY/dy."""
        (x, y) = xy
        _dYa_dy = 3*exp(-x)*y**2
        return _dYa_dy

    @staticmethod
    def d2Ya_dx2(xy):
        """Analytical d2Y/dx2."""
        (x, y) = xy
        _d2Ya_dx2 = exp(-x)*(-2 + x + y**3)
        return _d2Ya_dx2

    @staticmethod
    def d2Ya_dy2(xy):
        """Analytical d2Y/dy2."""
        (x, y) = xy
        _d2Ya_dy2 = 6*exp(-x)*y
        return _d2Ya_dy2

eq = Equation()

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
    # print("Python file: %s" % __file__)

def create_output_directory(path=None):
    path_noext, ext = os.path.splitext(path)
    output_dir = path_noext
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

def prod(n: list) -> int:
    """Compute the product of the elements of a list of numbers."""
    p = 1
    for nn in n:
        p *= nn
    return p

def create_training_grid2(*shape):
    """Create a grid of training data."""

    # Determine the number of dimensions in the result.
    m = len(shape)

    # Handle 1-D and (n>1)-D cases differently.
    if m == 1:
        n = shape[0]
        X = [i/(n - 1) for i in range(n)]
    else:
        # Compute the evenly-spaced points along each dimension.
        x = [[i/(n - 1) for i in range(n)] for n in shape]

        # Assemble all possible point combinations.
        X = []
        p1 = None
        p2 = 1
        for j in range(m - 1):
            p1 = prod(shape[j + 1:])
            XX = [xx for item in x[j] for xx in repeat(item, p1)]*p2
            X.append(XX)
            p2 *= shape[j]
        X.append(x[-1]*p2)
        X = list(zip(*X))

    # Return the list of training points.
    return X

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

print_system_information()

# Define the hyperparameters.

# Set up the output directory.
path = "./lagaris_05_tf_adam"
output_dir = create_output_directory(path)

# Training optimizer
training_algorithm = "Adam"

# Initial parameter ranges
w0_range = [-0.1, 0.1]
u0_range = [-0.1, 0.1]
v0_range = [-0.1, 0.1]

# Number of hidden nodes.
H = 10

# Number of training points in each dimension.
nx_train = 10
ny_train = 10
n_train = nx_train*ny_train

# Number of training epochs.
n_epochs = 16000

# Learning rate.
learning_rate = 0.01

# Random number generator seed.
random_seed = 0

# Relative tolerance for consecutive loss function values to indicate convergence.
tol = 1e-6

# Create and save the training data.
xy_train = create_training_data(nx_train, ny_train)
x_train = xy_train[::ny_train, 0]
y_train = xy_train[:ny_train, 1]
np.savetxt(os.path.join(output_dir,'xy_train.dat'), xy_train)

# Build the model.
model = build_model(H, w0_range, u0_range, v0_range)

# Create the optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Train the model.

# Create history variables.
losses = []
phist = []

# Set the random number seed for reproducibility.
tf.random.set_seed(random_seed)

# Rename the training data Variable (_v) for convenience, just for training.
xy_train_v = tf.Variable(xy_train, dtype=tf.float32, name="xy_train")
xy = xy_train_v

print("Hyperparameters: n_train = %s, H = %s, n_epochs = %s, learning_rate = %s"
      % (n_train, H, n_epochs, learning_rate))
t_start = datetime.datetime.now()
print("Training started at", t_start)

converged = False

for epoch in range(n_epochs):
    if epoch % 100 == 0:
        print("Starting epoch %d." % epoch)

    # Run the forward pass.
    with tf.GradientTape(persistent=True) as tape3:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:

                # Extract individual coordinates.
                x = xy[:, 0]
                y = xy[:, 1]

                # Compute the network output.
                N = model(xy)

                # Compute the trial solution.
                # NOTE:
                # This is (corrected) Eq. 23 in Lagaris, 1998.
                # The original equation has a - in front of the last term,
                # which is wrong.
                A = (
                    (1 - x)*y**3
                    + x*(1 + y**3)*tf.math.exp(-1.0)
                    + (1 - y)*x*(tf.math.exp(-x) - tf.math.exp(-1.0))
                    + y*(tf.math.exp(-x)*(x + 1) - (1 - x + 2*x*tf.math.exp(-1.0)))
                )
                P = x*(1 - x)*y*(1 - y)
                Y = A + P*N[:, 0]

            # Compute the gradient of the trial solution wrt inputs.
            dY_dx = tape1.gradient(Y, x)
            dY_dy = tape1.gradient(Y, y)

        # Compute the Laplacian of trial solution wrt inputs.
        d2Y_dx2 = tape2.gradient(dY_dx, x)
        d2Y_dy2 = tape2.gradient(dY_dy, y)

        # Compute the estimates of the differential equations.
        G = d2Y_dx2 + d2Y_dy2 - tf.math.exp(-x)*(x - 2 + y**3 + 6*y)

        # Compute the loss function.
        L = tf.math.sqrt(tf.reduce_sum(G**2)/n_train)

    # Save the current losses.
    losses.append(L.numpy())

    # Check for convergence.
    if epoch > 0:
        loss_delta = (losses[-1] - losses[-2])/losses[-2]
        if abs(loss_delta) <= tol:
            converged = True
            break

    # Compute the gradient of the loss function wrt the network parameters.
    pgrad = tape3.gradient(L, model.trainable_variables)

    # Save the parameters used in this pass.
    phist.append(
        np.hstack(
            (model.trainable_variables[0].numpy().reshape((2*H,)),    # w (2, H) matrix -> (2H,) row vector
             model.trainable_variables[1].numpy(),       # u (H,) row vector
             model.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )

    # Update the parameters for this pass.
    optimizer.apply_gradients(zip(pgrad, model.trainable_variables))

    
# Save the parameters used in the last pass.
phist.append(
    np.hstack(
        (model.trainable_variables[0].numpy().reshape((2*H,)),    # w (2, H) matrix -> (2H,) row vector
         model.trainable_variables[1].numpy(),       # u (H,) row vector
         model.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
    )
)

t_stop = datetime.datetime.now()
print("Training stopped at", t_stop)
t_elapsed = t_stop - t_start
print("Total training time was %s seconds." % t_elapsed.total_seconds())
print("Epochs: %d" % (epoch + 1))
print("Final value of loss function: %f" % losses[-1])
print("converged = %s" % converged)

# Save the parameter histories.
np.savetxt(os.path.join(output_dir, 'phist.dat'), np.array(phist))
