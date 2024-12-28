import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapezoid

from  Wake_Readout import total_wake, static_wake
from Airfoil_Readout import upper_taps, lower_taps, tap_coords

general = pd.read_csv('general_data.csv')

x_u = tap_coords()[0]*1.6/1000
x_l = tap_coords()[1]*1.6/1000

c_n_list = []
c_m_le_list = []
c_m_fourth_list = []

print(x_l[26])

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

def C_m_le(c_p_u, c_p_l):
    for i in range(0, len(c_p_u)):
        c_p_u[i] = c_p_u[i]*x_u[i]

    for i in range(0, len(c_p_l)):
        c_p_l[i] = c_p_l[i]*x_l[i+25]

    print(c_p_u)

    c_m_l = (trapezoid(c_p_u, x_u) - trapezoid(c_p_l, x_l))/0.160



    return c_m_l



for i in range(len(general['Alpha'])):

    c_p_u = C_p(general['Alpha'][i])[0]
    c_p_l = C_p(general['Alpha'][i])[1]
    

    
    c_n = (trapezoid(c_p_l,x_l) - trapezoid(c_p_u,x_u))/0.160

    c_m_le = C_m_le(c_p_u, c_p_l)
    c_m_fourth = c_m_le + 0.25 * c_n


    c_n_list.append(c_n)
    c_m_le_list.append(c_m_le)
    c_m_fourth_list.append(c_m_fourth)


desired_alpha = 6

c_p_u = C_p(desired_alpha)[0]
c_p_l = C_p(desired_alpha)[1]

# #Plotting Cp for a desired alpha
# plt.plot(x_u, c_p_u, marker='o', label="Upper Surface")
# plt.plot(x_l, c_p_l, marker='o', label="Lower Surface")
# plt.xlabel('Position (x [mm])')
# plt.ylabel('Cp, Pressure Coefficient')
# plt.title(f'Pressure Coefficient Distribution for Angle of Attack = {desired_alpha}°')

# plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
# plt.legend()
# plt.show()



# #Plotting Cn over alpha
plt.plot(general['Alpha'], c_n_list, label="C_n")
plt.plot(general['Alpha'], c_m_le_list, label="C_m_le")
plt.plot(general['Alpha'], c_m_fourth_list, label="C_m_c/4")
plt.legend()
plt.grid(True)
plt.show()