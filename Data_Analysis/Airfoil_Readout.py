import pandas as pd
import matplotlib.pyplot as plt


def tap_coords():
    coordinates_df = pd.read_csv('SLT_airfoil_taps_coords.csv', skiprows=1)
    # Map pressure readings to coordinates
    tap_ids = coordinates_df['pnum']

    x_positions = coordinates_df['x [%]'].astype(float)
    x_positions_upper = x_positions.loc[0:24] # Average the readings if needed
    x_positions_lower = x_positions.loc[25:49] # Average the readings if needed
    return x_positions_upper, x_positions_lower


def upper_taps(alpha):

    # Load the CSV data
    

    pressure_df = pd.read_csv('raw_2D.csv', skiprows=0)
    pressure_df = pressure_df.astype(float)


    # Select an angle of attack (Alpha)
    desired_alpha = alpha  # Change this to the angle of attack you want
    filtered_df = pressure_df[pressure_df['Alpha'] == desired_alpha]


    pressure_upper = filtered_df.loc[:, 'P001':'P025'].mean(axis=0)  # Average the readings if needed

    return pressure_upper
    


def lower_taps(alpha):
        # Load the CSV data
    coordinates_df = pd.read_csv('SLT_airfoil_taps_coords.csv', skiprows=1)

    pressure_df = pd.read_csv('raw_2D.csv', skiprows=0)
    pressure_df = pressure_df.astype(float)


    # Select an angle of attack (Alpha)
    desired_alpha = alpha  # Change this to the angle of attack you want
    filtered_df = pressure_df[pressure_df['Alpha'] == desired_alpha]

    # Map pressure readings to coordinates
    tap_ids = coordinates_df['pnum']

    x_positions = coordinates_df['x [%]'].astype(float)

    pressure_lower = filtered_df.loc[:, 'P026':'P049'].mean(axis=0)  # Average the readings if needed

    return pressure_lower


# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(x_positions_upper, pressure_upper, marker='o', label="Upper Surface")
# plt.plot(x_positions_lower, pressure_lower, marker='o', label="Lower Surface")
# plt.title(f'Pressure Distribution for Angle of Attack = {desired_alpha}°')
# plt.xlabel('Position (x [%])')
# plt.ylabel('Pressure (Pa)')
# plt.grid(True)
# plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
# plt.legend()
# plt.show()