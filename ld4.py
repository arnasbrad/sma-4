import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants for the problem
m1 = 0.5  # mass of object 1
m2 = 0.25  # mass of object 2
k1 = 0.02  # air resistance coefficient of object 1
k2 = 0.04  # air resistance coefficient of object 2
ks = 0.002  # air resistance coefficient when objects are combined
v0 = 100.0  # initial velocity
ts = 2.0  # time when objects start moving separately
t_max = 15.0  # maximum simulation time
g = 9.81  # acceleration due to gravity

# Corrected motion function with air resistance
def motion(t, y, m, k):
    dvdt = -g - (k * y[1] ** 2 * np.sign(y[1])) / m
    return [y[1], dvdt]


# Euler method implementation
def euler_method(f, t_span, y0, args, steps):
    t0, tf = t_span
    h = (tf - t0) / steps
    t_values = np.linspace(t0, tf, steps + 1)
    y_values = np.zeros((len(y0), steps + 1))
    y_values[:, 0] = y0

    for i in range(steps):
        y_values[:, i + 1] = y_values[:, i] + h * np.array(
            f(t_values[i], y_values[:, i], *args)
        )

    return t_values, y_values


# Runge-Kutta 4th order method implementation
def runge_kutta_4th_order(f, t_span, y0, args, steps):
    t0, tf = t_span
    h = (tf - t0) / steps
    t_values = np.linspace(t0, tf, steps + 1)
    y_values = np.zeros((len(y0), steps + 1))
    y_values[:, 0] = y0

    for i in range(steps):
        k1 = np.array(f(t_values[i], y_values[:, i], *args))
        k2 = np.array(f(t_values[i] + h / 2, y_values[:, i] + h / 2 * k1, *args))
        k3 = np.array(f(t_values[i] + h / 2, y_values[:, i] + h / 2 * k2, *args))
        k4 = np.array(f(t_values[i] + h, y_values[:, i] + h * k3, *args))

        y_values[:, i + 1] = y_values[:, i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, y_values


# Function to perform the simulations with different step sizes
def simulate_motion(steps_count):
    # Solving using Euler method for combined motion
    t_euler_combined, y_euler_combined = euler_method(
        motion, [0, ts], [0, v0], args=(m1 + m2, ks), steps=steps_count
    )

    # Initial conditions for separate motions using Euler's results
    v1_init_euler = y_euler_combined[1, -1]
    v2_init_euler = y_euler_combined[1, -1]

    # Solving using Euler method for separate motions
    t_euler1, y_euler1 = euler_method(
        motion,
        [ts, t_max],
        [y_euler_combined[0, -1], v1_init_euler],
        args=(m1, k1),
        steps=steps_count,
    )
    t_euler2, y_euler2 = euler_method(
        motion,
        [ts, t_max],
        [y_euler_combined[0, -1], v2_init_euler],
        args=(m2, k2),
        steps=steps_count,
    )

    # Solving using Runge-Kutta 4th order method for combined motion
    t_rk4_combined, y_rk4_combined = runge_kutta_4th_order(
        motion, [0, ts], [0, v0], args=(m1 + m2, ks), steps=steps_count
    )

    # Initial conditions for separate motions using RK4's results
    v1_init_rk4 = y_rk4_combined[1, -1]
    v2_init_rk4 = y_rk4_combined[1, -1]

    # Solving using Runge-Kutta 4th order method for separate motions
    t_rk41, y_rk41 = runge_kutta_4th_order(
        motion,
        [ts, t_max],
        [y_rk4_combined[0, -1], v1_init_rk4],
        args=(m1, k1),
        steps=steps_count,
    )
    t_rk42, y_rk42 = runge_kutta_4th_order(
        motion,
        [ts, t_max],
        [y_rk4_combined[0, -1], v2_init_rk4],
        args=(m2, k2),
        steps=steps_count,
    )

    # Plotting results from Euler method
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_euler_combined, y_euler_combined[0], label="Combined motion (Euler)")
    plt.plot(t_euler1, y_euler1[0], label="Object 1 motion (Euler)", linestyle="--")
    plt.plot(t_euler2, y_euler2[0], label="Object 2 motion (Euler)", linestyle="-.")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title(f"Euler Method: Height vs Time (Step Size: {steps_count})")
    plt.legend()
    plt.grid(True)

    # Plotting results from Runge-Kutta 4th order method
    plt.subplot(1, 2, 2)
    plt.plot(t_rk4_combined, y_rk4_combined[0], label="Combined motion (RK4)")
    plt.plot(t_rk41, y_rk41[0], label="Object 1 motion (RK4)", linestyle="--")
    plt.plot(t_rk42, y_rk42[0], label="Object 2 motion (RK4)", linestyle="-.")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title(
        f"Runge-Kutta 4th Order Method: Height vs Time (Step Size: {steps_count})"
    )
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Function to compare our methods with scipy.integrate's solve_ivp


def compare_with_solve_ivp(step_size):

    # Define the time spans for the combined and separate motions

    time_span_combined = (0, ts)

    time_span_separate = (ts, t_max)

    # Solve with solve_ivp for combined motion

    sol_combined = solve_ivp(
        lambda t, y: motion(t, y, m1 + m2, ks),
        time_span_combined,
        [0, v0],
        t_eval=np.linspace(0, ts, step_size),
    )

    # Initial conditions for separate motions using solve_ivp's results

    v1_init_ivp = sol_combined.y[1, -1]

    v2_init_ivp = sol_combined.y[1, -1]

    # Solve with solve_ivp for separate motions

    sol1 = solve_ivp(
        lambda t, y: motion(t, y, m1, k1),
        time_span_separate,
        [sol_combined.y[0, -1], v1_init_ivp],
        t_eval=np.linspace(ts, t_max, step_size),
    )

    sol2 = solve_ivp(
        lambda t, y: motion(t, y, m2, k2),
        time_span_separate,
        [sol_combined.y[0, -1], v2_init_ivp],
        t_eval=np.linspace(ts, t_max, step_size),
    )

    # Plotting results from solve_ivp

    plt.figure(figsize=(15, 5))

    plt.plot(
        sol_combined.t,
        sol_combined.y[0],
        label="Combined motion (solve_ivp)",
        color="orange",
    )

    plt.plot(
        sol1.t,
        sol1.y[0],
        label="Object 1 motion (solve_ivp)",
        linestyle="--",
        color="orange",
    )

    plt.plot(
        sol2.t,
        sol2.y[0],
        label="Object 2 motion (solve_ivp)",
        linestyle="-.",
        color="orange",
    )

    plt.xlabel("Time (s)")

    plt.ylabel("Height (m)")

    plt.title(f"solve_ivp Method: Height vs Time (Step Size: {step_size})")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# Simulate with different step sizes
steps_counts = [100, 250, 500]
for count in steps_counts:
    simulate_motion(count)

compare_with_solve_ivp(steps_counts[-1])  # Using the largest step size for comparison
