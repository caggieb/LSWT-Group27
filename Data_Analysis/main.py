import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapezoid

from  Wake_Readout import total_wake, static_wake
from Airfoil_Readout import upper_taps, lower_taps, tap_coords

general = pd.read_csv('general_data.csv')

x_u = tap_coords()[0]
x_l = tap_coords()[1]

c_n_list = []

def C_p(alpha):

    desired_alpha = alpha

    general_alpha = general[general['Alpha'] == desired_alpha]

    p_stat = general_alpha.iloc[0]['p_stat']
    q_inf = general_alpha.iloc[0]['q_inf']

    p_u = upper_taps(desired_alpha)
    p_l = lower_taps(desired_alpha)

    c_p_u = (p_u - p_stat)/q_inf
    c_p_l = (p_l - p_stat)/q_inf

    return c_p_u, c_p_l

for i in range(len(general['Alpha'])):

    c_p_u = C_p(general['Alpha'][i])[0]
    c_p_l = C_p(general['Alpha'][i])[1]
    
    c_n = (trapezoid(c_p_l) - trapezoid(c_p_u))/100

    c_n_list.append(c_n)

plt.plot(general['Alpha'], c_n_list,)
# plt.plot(x_u, c_p_u, marker='o', label="Upper Surface")
# plt.plot(x_l, c_p_l, marker='o', label="Lower Surface")
# plt.xlabel('Position (x [%])')
# plt.ylabel('Cp, Pressure Coefficient')
# plt.title(f'Pressure Coefficient Distribution for Angle of Attack = {desired_alpha}°')
plt.grid(True)
# plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
# plt.legend()
plt.show()