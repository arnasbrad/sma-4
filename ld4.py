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
    dvdt = -g - (k * y[1]**2 * np.sign(y[1])) / m
    return [y[1], dvdt]

# Euler method implementation
def euler_method(f, t_span, y0, args, steps):
    t0, tf = t_span
    h = (tf - t0) / steps
    t_values = np.linspace(t0, tf, steps + 1)
    y_values = np.zeros((len(y0), steps + 1))
    y_values[:, 0] = y0

    for i in range(steps):
        y_values[:, i + 1] = y_values[:, i] + h * np.array(f(t_values[i], y_values[:, i], *args))
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

# Function to find the time when the object reaches the highest point (velocity = 0)
def find_highest_point_time(t_values, y_values):
    idx = np.where(y_values[1] <= 0)[0][0] 
    return t_values[idx]

# Function to perform the simulations with different step sizes and plot speed over time
def simulate_speed_over_time(steps_count):
    # Solving using Euler method for combined motion
    t_euler_combined, y_euler_combined = euler_method(motion, [0, ts], [0, v0], args=(m1 + m2, ks), steps=steps_count)

    # Solving using Euler method for separate motions
    t_euler1, y_euler1 = euler_method(motion, [ts, t_max], [y_euler_combined[0, -1], y_euler_combined[1, -1]], args=(m1, k1), steps=steps_count)
    t_euler2, y_euler2 = euler_method(motion, [ts, t_max], [y_euler_combined[0, -1], y_euler_combined[1, -1]], args=(m2, k2), steps=steps_count)

    # Solving using Runge-Kutta 4th order method for combined motion
    t_rk4_combined, y_rk4_combined = runge_kutta_4th_order(motion, [0, ts], [0, v0], args=(m1 + m2, ks), steps=steps_count)

    # Solving using Runge-Kutta 4th order method for separate motions
    t_rk41, y_rk41 = runge_kutta_4th_order(motion, [ts, t_max], [y_rk4_combined[0, -1], y_rk4_combined[1, -1]], args=(m1, k1), steps=steps_count)
    t_rk42, y_rk42 = runge_kutta_4th_order(motion, [ts, t_max], [y_rk4_combined[0, -1], y_rk4_combined[1, -1]], args=(m2, k2), steps=steps_count)

    # Plotting results for speed over time
    plt.figure(figsize=(15, 10))

    # Plotting results from Euler method
    plt.subplot(2, 1, 1)
    plt.plot(t_euler_combined, y_euler_combined[1], label="Combined motion (Euler)")
    plt.plot(t_euler1, y_euler1[1], label="Object 1 motion (Euler)", linestyle="--")
    plt.plot(t_euler2, y_euler2[1], label="Object 2 motion (Euler)", linestyle="-.")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Euler Method: Speed vs Time (Step Size: {steps_count})")
    plt.legend()
    plt.grid(True)

    # Plotting results from Runge-Kutta 4th order method
    plt.subplot(2, 1, 2)
    plt.plot(t_rk4_combined, y_rk4_combined[1], label="Combined motion (RK4)")
    plt.plot(t_rk41, y_rk41[1], label="Object 1 motion (RK4)", linestyle="--")
    plt.plot(t_rk42, y_rk42[1], label="Object 2 motion (RK4)", linestyle="-.")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Runge-Kutta 4th Order Method: Speed vs Time (Step Size: {steps_count})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Finding the time each object reaches the highest point
    time_highest_euler1 = find_highest_point_time(t_euler1, y_euler1)
    time_highest_euler2 = find_highest_point_time(t_euler2, y_euler2)
    time_highest_rk41 = find_highest_point_time(t_rk41, y_rk41)
    time_highest_rk42 = find_highest_point_time(t_rk42, y_rk42)

    return time_highest_euler1, time_highest_euler2, time_highest_rk41, time_highest_rk42

def simulate_with_solve_ivp():
    # First simulate the combined motion
    sol_combined = solve_ivp(lambda t, y: motion(t, y, m1 + m2, ks), [0, ts], [0, v0], t_eval=np.linspace(0, ts, 250))

    # Now simulate the separate motions using the final state of the combined motion as the initial state
    v1_init_ivp = sol_combined.y[1, -1]  # Final velocity of the combined motion for object 1
    v2_init_ivp = sol_combined.y[1, -1]  # Final velocity of the combined motion for object 2

    sol1 = solve_ivp(lambda t, y: motion(t, y, m1, k1), [ts, t_max], [sol_combined.y[0, -1], v1_init_ivp], t_eval=np.linspace(ts, t_max, 250))
    sol2 = solve_ivp(lambda t, y: motion(t, y, m2, k2), [ts, t_max], [sol_combined.y[0, -1], v2_init_ivp], t_eval=np.linspace(ts, t_max, 250))

    # Combine the t and y arrays for the complete motion
    t_full = np.concatenate((sol_combined.t, sol1.t))
    y_full_speed1 = np.concatenate((sol_combined.y[1], sol1.y[1]))
    y_full_speed2 = np.concatenate((sol_combined.y[1], sol2.y[1]))  # Note: using sol_combined.y[1] because it's the same initial velocity for both objects

    # Plotting results from solve_ivp
    plt.figure(figsize=(15, 5))
    plt.plot(t_full, y_full_speed1, label="Object 1 motion (solve_ivp)", color="orange")
    plt.plot(t_full, y_full_speed2, label="Object 2 motion (solve_ivp)", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("solve_ivp Method: Speed vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulate speed over time with different step sizes and find highest point times
steps_counts = [100, 250, 500]
for count in steps_counts:
    times_highest = simulate_speed_over_time(count)
    print(f"Step Size: {count}")
    print(f"Time highest point (Euler) - Object 1: {times_highest[0]:.2f} s, Object 2: {times_highest[1]:.2f} s")
    print(f"Time highest point (RK4) - Object 1: {times_highest[2]:.2f} s, Object 2: {times_highest[3]:.2f} s\n")

# Compare with solve_ivp
simulate_with_solve_ivp()

