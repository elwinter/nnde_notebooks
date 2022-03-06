# Import standard Python modules.
import datetime
import os
import platform
import sys

# Import 3rd-party modules.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import TensorFlow.
import tensorflow as tf


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
    x_train = np.linspace(0, 1, n_train[0])
    return x_train


def build_model(H, w0_range, u0_range, v0_range):
    hidden_layer_1 = tf.keras.layers.Dense(
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
    model = tf.keras.Sequential([hidden_layer_1, output_layer])
    return model


# Define the differential equation using TensorFlow operations.
@tf.function
def ode_u(x, u, du_dx):
    G = du_dx - (2*x + 1)**-0.5
    return G


# Define the analytical solution and derivative.
u_analytical = lambda x: (2*x + 1)**0.5
du_dx_analytical = lambda x: (2*x + 1)**-0.5


# Define the trial function using TensorFlow operations.
@tf.function
def Y_trial_u(x, N):
    A = tf.constant([[1.0]], dtype="float64")
    P = x
    Y = A + P*N
    return Y


def plot_analytical_solution(output_dir, eq_name):
    """Plot the analytical solution to a file."""
    # Compute the analytical solution and derivative.
    nx = 101
    xa = np.linspace(0, 1, nx)
    ua = u_analytical(xa)
    dua_dx = du_dx_analytical(xa)

    # Plot the analytical solution to a file.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xa, ua, label="$u_a$")
    ax.plot(xa, dua_dx, label="$du_a/dx$")
    ax.set_xlabel("x")
    ax.set_ylabel("$u_a$ or $du_a/dx$")
    ax.grid()
    ax.legend()
    ax.set_title("Analytical solution and derivative for %s" % eq_name)
    path = os.path.join(output_dir, "analytical.png")
    plt.savefig(path)


def plot_loss_function_history(output_dir, eq_name, losses):
    """Plot the loss function history."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss function (RMS error)")
    ax.grid()
    ax.set_title("Loss function evolution for %s" % eq_name)
    path = os.path.join(output_dir, "loss.png")
    plt.savefig(path)


def plot_trained_solution(output_dir, eq_name, x_train, ut_train, dut_dx_train):
    """Plot the trained solution."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_train, ut_train, label="$u_t$")
    ax.plot(x_train, dut_dx_train, label="$du_t/dx$")
    ax.set_xlabel("x")
    ax.set_ylabel("$u_t$ or $du_t/dx$")
    ax.grid()
    ax.legend()
    ax.set_title("Trained solution and derivative for %s" % eq_name)
    path = os.path.join(output_dir, "trained.png")
    plt.savefig(path)


def plot_trained_error(output_dir, eq_name, x_train, ut_err_train, dut_dx_err_train):
    """Plot the trained solution."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_train, ut_err_train, label="$u_t - u_a$")
    ax.plot(x_train, dut_dx_err_train, label="$du_t/dx - du_a/dx$")
    ax.set_xlabel("x")
    ax.set_ylabel("Trained - analytical")
    ax.grid()
    ax.legend()
    ax.set_title("Error in trained solution for %s" % eq_name)
    path = os.path.join(output_dir, "error.png")
    plt.savefig(path)


def plot_parameter_history(output_dir, eq_name, phist, H, m):
    """Plot the parameter history."""
    phist = np.array(phist)
    plt.figure(figsize=(12, 14))

    # w
    plt.subplot(311)
    plt.plot(phist[:, 0:H])
    plt.title("Hidden node weight $w$")
    plt.grid()

    # u
    plt.subplot(312)
    plt.plot(phist[:, H:2*H])
    plt.title("Hidden node bias $u$")
    plt.grid()

    # v
    plt.subplot(313)
    plt.plot(phist[:, 2*H:3*H])
    plt.title("Output node weight $v$")
    plt.grid()

    plt.suptitle("Parameter evolution for %s" % eq_name)
    plt.subplots_adjust(hspace=0.2)
    path = os.path.join(output_dir, "parameters.png")
    plt.savefig(path)


if __name__ == "__main__":

    # Use 64-bit math in TensorFlow.
    tf.keras.backend.set_floatx("float64")

    # Define the run parameters.
    eq_name = "case1"
    m = 1               # Number of dimensions
    nx_train = 11       # Number of training points in x-dimension
    n_train = nx_train  # Total number of training points.
    random_seed = 0     # Random number generator seed

    # Define the hyperparameters for the neural network.
    optimizer_name = "Adam"  # Training optimizer
    w0_range = [-0.1, 0.1]   # Initial w range
    u0_range = [-0.1, 0.1]   # Initial u range
    v0_range = [-0.1, 0.1]   # Initial v range
    max_epochs = 100         # Maximum number of training epochs
    learning_rate = 0.01     # Learning rate.
    tol = 1e-6               # Absolute tolerance for loss function convergence
    H = 10                   # Number of hidden nodes per layer

    # Print the system description.
    print_system_information()

    # Set up the output directory.
    path = os.path.join(".", eq_name)
    output_dir = create_output_directory(path)

    # Create plots in buffers for writing to disk.
    matplotlib.use("Agg")

    # Plot the analytical solution and derivative.
    plot_analytical_solution(output_dir, eq_name)

    # Create and save the training data.
    x_train = create_training_data(nx_train)
    np.savetxt(os.path.join(output_dir, "x_train.dat"), x_train)

    # Build the model.
    model_u = build_model(H, w0_range, u0_range, v0_range)

    # Create the optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create history variables.
    losses_u = []
    losses = []
    phist_u = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(random_seed)

    # Rename the training data Variable for convenience, just for training.
    # shape (n_train, m)
    x_train_var = tf.Variable(np.array(x_train).reshape((n_train, m)), name="x_train")
    x = x_train_var

    # Clear the convergence flag to start.
    converged = False

    # Train the model.
    print("Hyperparameters: n_train = %s, H = %s, max_epochs = %s, optimizer = %s, learning_rate = %s"
        % (n_train, H, max_epochs, optimizer_name, learning_rate))
    t_start = datetime.datetime.now()
    print("Training started at", t_start)

    for epoch in range(max_epochs):

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape1:
            with tf.GradientTape(persistent=True) as tape0:

                # Compute the network outputs at the training points.
                N_u = model_u(x)

                # Compute the trial solutions.
                u = Y_trial_u(x, N_u)

            # Compute the gradients of the trial solutions wrt inputs.
            du_dx = tape0.gradient(u, x)

            # Compute the estimates of the differential equations.
            G_u = ode_u(x, u, du_dx)

            # Compute the loss functions.
            L_u = tf.math.sqrt(tf.reduce_sum(G_u**2)/n_train)
            L = L_u

        # Save the current losses.
        losses_u.append(L_u.numpy())
        losses.append(L.numpy())

        # Check for convergence.
        if epoch > 0:
            loss_delta = losses[-1] - losses[-2]
            if abs(loss_delta) <= tol:
                converged = True
                break

        # Compute the gradient of the loss function wrt the network parameters.
        pgrad_u = tape1.gradient(L, model_u.trainable_variables)

        # Save the parameters used in this epoch.
        phist_u.append(
            np.hstack(
                (model_u.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
                model_u.trainable_variables[1].numpy(),       # u (H,) row vector
                model_u.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )

        # Update the parameters for this epoch.
        optimizer.apply_gradients(zip(pgrad_u, model_u.trainable_variables))

        if epoch % 100 == 0:
            print("Ending epoch %s, loss function = %f" % (epoch, L.numpy()))

    # Save the parameters used in the last epoch.
    phist_u.append(
        np.hstack(
            (model_u.trainable_variables[0].numpy().reshape((m*H,)),    # w (m, H) matrix -> (m*H,) row vector
            model_u.trainable_variables[1].numpy(),       # u (H,) row vector
            model_u.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
        )
    )

    # Increment epoch count for last epoch.
    n_epochs = epoch + 1

    # Print end report.
    t_stop = datetime.datetime.now()
    print("Training stopped at", t_stop)
    t_elapsed = t_stop - t_start
    print("Total training time was %s seconds." % t_elapsed.total_seconds())
    print("Epochs: %d" % n_epochs)
    print("Final value of loss function: %f" % losses[-1])
    print("converged = %s" % converged)

    # Save the parameter and loss function histories.
    np.savetxt(os.path.join(output_dir, 'phist_u.dat'), np.array(phist_u))
    np.savetxt(os.path.join(output_dir, 'losses.dat'), np.array(losses))

    # Compute and save the trained results at training points.
    with tf.GradientTape(persistent=True) as tape:
        N_u = model_u(x)
        ut_train = Y_trial_u(x, N_u)
    dut_dx_train = tape.gradient(ut_train, x)
    np.savetxt(os.path.join(output_dir, "ut_train.dat"), ut_train.numpy().reshape((n_train,)))
    np.savetxt(os.path.join(output_dir, "dut_dx_train.dat"), dut_dx_train.numpy())

    # Compute and save the analytical solution and derivative at training points.
    ua_train = u_analytical(x_train)
    dua_dx_train = du_dx_analytical(x_train)
    np.savetxt(os.path.join(output_dir,"ua_train.dat"), ua_train)
    np.savetxt(os.path.join(output_dir,"dua_dx_train.dat"), dua_dx_train)

    # Compute and save the error in the trained solution and derivative at training points.
    ut_err_train = ut_train.numpy().reshape((nx_train,)) - ua_train
    dut_dx_err_train = dut_dx_train.numpy().reshape((nx_train,)) - dua_dx_train
    np.savetxt(os.path.join(output_dir, "ut_err_train.dat"), ut_err_train)
    np.savetxt(os.path.join(output_dir, "dut_dx_err_train.dat"), dut_dx_err_train)

    # Compute the final RMS error in the solution at the training points.
    ut_rmse_train = np.sqrt(np.sum(ut_err_train**2)/n_train)
    print("ut_rmse_train = %s" % ut_rmse_train)

    # Plot the loss function history.
    plot_loss_function_history(output_dir, eq_name, losses)

    # Plot the the trained solution and derivative at the training points.
    plot_trained_solution(output_dir, eq_name, x_train, ut_train, dut_dx_train)

    # Plot the errors in the trained solution and derivative at the training points.
    plot_trained_error(output_dir, eq_name, x_train, ut_err_train, dut_dx_err_train)

    # Plot the parameter histories.
    plot_parameter_history(output_dir, eq_name, phist_u, H, m)
