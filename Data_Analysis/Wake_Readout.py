import pandas as pd
import matplotlib.pyplot as plt

def total_wake(alpha):

    # Load the CSV data
    total_coordinates_df = pd.read_csv('SLT_total_rake_coords.csv', skiprows=1)

    static_coordinates_df = pd.read_csv('SLT_static_rake_coords.csv', skiprows=1)

    pressure_df = pd.read_csv('raw_2D.csv', skiprows=0)
    pressure_df = pressure_df.astype(float)



    # Select an angle of attack (Alpha)
    desired_alpha = alpha  # Change this to the angle of attack you want
    filtered_df = pressure_df[pressure_df['Alpha'] == desired_alpha]

    # Map pressure readings to coordinates
    tap_ids = total_coordinates_df['pnum']

    x_positions_total = total_coordinates_df['pos'].astype(float)

    pressure_total = filtered_df.loc[:, 'P050':'P096'].mean(axis=0)  # Average the readings if needed
    
    return pressure_total, x_positions_total


def static_wake(alpha):
    # Load the CSV data
    total_coordinates_df = pd.read_csv('SLT_total_rake_coords.csv', skiprows=1)

    static_coordinates_df = pd.read_csv('SLT_static_rake_coords.csv', skiprows=1)

    pressure_df = pd.read_csv('raw_2D.csv', skiprows=0)
    pressure_df = pressure_df.astype(float)



    # Select an angle of attack (Alpha)
    desired_alpha = alpha  # Change this to the angle of attack you want
    filtered_df = pressure_df[pressure_df['Alpha'] == desired_alpha]

    x_positions_static = static_coordinates_df['pos'].astype(float)

    pressure_static = filtered_df.loc[:, 'P098':'P109'].mean(axis=0)  # Average the readings if needed

    return pressure_static, x_positions_static

desired_alpha = 6

pressure_static, x_positions_static = static_wake(desired_alpha)
pressure_total, x_positions_total = total_wake(desired_alpha)


# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(x_positions_total, pressure_total, marker='o', label="Total Rake Pressure")
# plt.plot(x_positions_static, pressure_static, marker='o', label="Static Rake Pressure")
# plt.title(f'Wake Rake Pressure Distribution for Angle of Attack = {desired_alpha}°')
# plt.xlabel('Relative Position ([mm])')
# plt.ylabel('Pressure (Pa)')
# plt.grid(True)
# plt.legend()
# plt.show()