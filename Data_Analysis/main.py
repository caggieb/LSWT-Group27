import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapezoid

from  Wake_Readout import total_wake, static_wake
from Airfoil_Readout import upper_taps, lower_taps, tap_coords
from Airfoil_Geom import read_airfoil_data, create_angle_functions

# Load airfoil data
filename = 'sd6060.dat'
x_airf_upper, y_airf_upper, x_airf_lower, y_airf_lower = read_airfoil_data(filename)

# Create angle functions
upper_surface_angle, lower_surface_angle = create_angle_functions(x_airf_upper, y_airf_upper, x_airf_lower, y_airf_lower)

# Plot airfoil with angles
#plot_airfoil_with_angles(x_upper, y_upper, x_lower, y_lower, upper_angle_func, lower_angle_func)

general = pd.read_csv('general_data.csv')

x_u = tap_coords()[0]*1.6/1000
x_l = tap_coords()[1]*1.6/1000

c_n_list = []
c_t_list = []
c_m_le_list = []
c_m_fourth_list = []
c_l_list = []
c_d_list = []

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
        c_p_u.iloc[i] = c_p_u.iloc[i]*x_u[i]

    for i in range(0, len(c_p_l)):
        c_p_l.iloc[i] = c_p_l.iloc[i]*x_l[i+25]

    c_m_l = (trapezoid(c_p_u, x_u) - trapezoid(c_p_l, x_l))/0.160

    return c_m_l



for i in range(len(general['Alpha'])):

    c_p_u = C_p(general['Alpha'][i])[0]
    c_p_l = C_p(general['Alpha'][i])[1]
    
    c_p_t_u = np.linspace(0,0, len(c_p_u))
    c_p_t_l = np.linspace(0,0, len(c_p_l))

    for i in range(0,len(c_p_u)):
        c_p_t_u[i] = c_p_u.iloc[i]*math.sin(upper_surface_angle(x_u.iloc[i]/(160/1000)))

    for i in range(0,len(c_p_l)):
        c_p_t_l[i] = c_p_l.iloc[i]*math.sin(lower_surface_angle(x_l.iloc[i]/(160/1000)))


    c_n = (trapezoid(c_p_l,x_l) - trapezoid(c_p_u,x_u))/0.160
    c_t = (trapezoid(c_p_t_u,x_u) - trapezoid(c_p_t_l,x_l))/0.160

    c_m_le = C_m_le(c_p_u, c_p_l)
    c_m_fourth = c_m_le + 0.25 * c_n

    aoa_rad = math.radians(general['Alpha'][i])

    c_l = c_n*math.cos(aoa_rad) - c_t*math.sin(aoa_rad)
    c_d = c_t*math.cos(aoa_rad) + c_n*math.sin(aoa_rad)

    c_n_list.append(c_n)
    c_t_list.append(c_t)
    c_m_le_list.append(c_m_le)
    c_m_fourth_list.append(c_m_fourth)
    c_l_list.append(c_l)
    c_d_list.append(c_d)


# for i in range(len(general['Alpha'])):

#     p_u = upper_taps(general['Alpha'][i])
#     p_l = lower_taps(general['Alpha'][i])
    
#     p_t_u = np.linspace(0,0, len(p_u))
#     p_t_l = np.linspace(0,0, len(p_l))

#     for i in range(0,len(p_u)):
#         p_t_u[i] = p_u.iloc[i]*math.sin(upper_surface_angle(x_u.iloc[i]/(160/1000)))

#     for i in range(0,len(p_l)):
#         p_t_l[i] = p_l.iloc[i]*math.sin(lower_surface_angle(x_l.iloc[i]/(160/1000)))


#     n_star = (trapezoid(p_l,x_l) - trapezoid(p_u,x_u))
#     a_star = (trapezoid(p_t_u,x_u) - trapezoid(p_t_l,x_l))

#     c_m_le = C_m_le(c_p_u, c_p_l)
#     c_m_fourth = c_m_le + 0.25 * c_n

#     aoa_rad = math.radians(general['Alpha'][i])

#     l_star = n_star*math.cos(aoa_rad) - a_star*math.sin(aoa_rad)
#     d_star = a_star*math.cos(aoa_rad) + n_star*math.sin(aoa_rad)



#     c_n_list.append(c_n)
#     c_t_list.append(c_t)
#     c_m_le_list.append(c_m_le)
#     c_m_fourth_list.append(c_m_fourth)
#     c_l_list.append(c_l)
#     c_d_list.append(c_d)

desired_alpha = 6

c_p_u = C_p(desired_alpha)[0]
c_p_l = C_p(desired_alpha)[1]


# #Plotting Cp chordwise for a desired alpha
# plt.plot(x_u, c_p_u, marker='o', label="Upper Surface")
# plt.plot(x_l, c_p_l, marker='o', label="Lower Surface")
# plt.xlabel('Position (x [mm])')
# plt.ylabel('Cp, Pressure Coefficient')
# plt.title(f'Pressure Coefficient Distribution for Angle of Attack = {desired_alpha}°')

# plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
# plt.legend()
# plt.show()



# Plotting Coeffs over alpha
plt.plot(general['Alpha'], c_n_list, label="C_n")
plt.plot(general['Alpha'], c_t_list, label="C_t")
#plt.plot(general['Alpha'], c_m_le_list, label="C_m_le")
#plt.plot(general['Alpha'], c_m_fourth_list, label="C_m_c/4")
#plt.plot(general['Alpha'], c_l_list, label="C_l")
plt.plot(general['Alpha'], c_d_list, label="C_d")
plt.xlabel('coeff')
plt.ylabel('alpha')
plt.legend()
plt.grid(True)
plt.show()