import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

# File reading function
def read_airfoil_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip the header (first two lines)
    data_lines = lines[2:]

    # Parse the data
    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []

    is_upper = True
    for line in data_lines:
        line = line.strip()
        if line == "":  # Empty line indicates switching between upper and lower surfaces
            continue

        x, y = map(float, line.split())
        if is_upper:
            if x in x_upper:  # Duplicate point indicates transition to lower surface
                is_upper = False
                x_lower.append(x)
                y_lower.append(y)
            else:
                x_upper.append(x)
                y_upper.append(y)
        else:
            x_lower.append(x)
            y_lower.append(y)

    return x_upper, y_upper, x_lower, y_lower

# Plotting function
def plot_airfoil_with_angles(x_upper, y_upper, x_lower, y_lower, upper_angle_func, lower_angle_func):
    plt.figure(figsize=(12, 8))

    # Plot upper and lower curves
    plt.plot(x_upper, y_upper, label="Upper Surface", color="blue")
    plt.plot(x_lower, y_lower, label="Lower Surface", color="red")

    # Calculate and plot angles
    angles_upper = [upper_angle_func(x) for x in x_upper]
    angles_lower = [lower_angle_func(x) for x in x_lower]

    plt.plot(x_upper, angles_upper, label="Upper Surface Angle", color="cyan", linestyle="--")
    plt.plot(x_lower, angles_lower, label="Lower Surface Angle", color="orange", linestyle="--")

    # Labels and title
    plt.title("Airfoil Shape with Surface Angles")
    plt.xlabel("Chord Position (x)")
    plt.ylabel("Thickness / Angle (radians)")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Add centerline
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()

# Function to calculate angles of upper and lower surfaces
def calculate_surface_angles(x_upper, y_upper, x_lower, y_lower):
    # Interpolate the surfaces
    upper_interp = interp1d(x_upper, y_upper, kind='cubic', fill_value="extrapolate")
    lower_interp = interp1d(x_lower, y_lower, kind='cubic', fill_value="extrapolate")

    def upper_angle(x):
        dy_dx = np.gradient(upper_interp(np.array([x - 1e-5, x, x + 1e-5]))) / np.gradient(np.array([x - 1e-5, x, x + 1e-5]))
        angle = np.arctan(dy_dx[1])
        return angle

    def lower_angle(x):
        dy_dx = np.gradient(lower_interp(np.array([x - 1e-5, x, x + 1e-5]))) / np.gradient(np.array([x - 1e-5, x, x + 1e-5]))
        angle = np.arctan(dy_dx[1])
        return angle

    return upper_angle, lower_angle

# Functions for external use
def create_angle_functions(x_upper, y_upper, x_lower, y_lower):
    upper_angle_func, lower_angle_func = calculate_surface_angles(x_upper, y_upper, x_lower, y_lower)
    
    def get_upper_surface_angle(x):
        return upper_angle_func(x)

    def get_lower_surface_angle(x):
        return lower_angle_func(x)

    return get_upper_surface_angle, get_lower_surface_angle

def create_surface_functions(x_upper, y_upper, x_lower, y_lower):
    upper_interp = interp1d(x_upper, y_upper, kind='cubic', fill_value="extrapolate")
    lower_interp = interp1d(x_lower, y_lower, kind='cubic', fill_value="extrapolate")

    def get_upper_surface_y(x):
        return upper_interp(x)

    def get_lower_surface_y(x):
        return lower_interp(x)

    return get_upper_surface_y, get_lower_surface_y

# Main script
if __name__ == "__main__":
    filename = 'sd6060.dat'
    x_upper, y_upper, x_lower, y_lower = read_airfoil_data(filename)

    # Create functions to calculate angles
    upper_angle_func, lower_angle_func = calculate_surface_angles(x_upper, y_upper, x_lower, y_lower)

    # Plot airfoil with angles
    plot_airfoil_with_angles(x_upper, y_upper, x_lower, y_lower, upper_angle_func, lower_angle_func)

    # Create callable angle and surface functions for external use
    get_upper_surface_angle, get_lower_surface_angle = create_angle_functions(x_upper, y_upper, x_lower, y_lower)
    get_upper_surface_y, get_lower_surface_y = create_surface_functions(x_upper, y_upper, x_lower, y_lower)