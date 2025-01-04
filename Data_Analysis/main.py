import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from  Wake_Readout import total_wake, static_wake
from Airfoil_Readout import upper_taps, lower_taps, tap_coords
from Airfoil_Geom import read_airfoil_data, create_angle_functions
from quickwrite import quickwrite

# Load airfoil data
filename = 'sd6060.dat'
x_airf_upper, y_airf_upper, x_airf_lower, y_airf_lower = read_airfoil_data(filename)

# Create angle functions
upper_surface_angle, lower_surface_angle = create_angle_functions(x_airf_upper, y_airf_upper, x_airf_lower, y_airf_lower)

# Plot airfoil with angles
#plot_airfoil_with_angles(x_upper, y_upper, x_lower, y_lower, upper_angle_func, lower_angle_func)

general = pd.read_csv('general_data.csv')

x_u = tap_coords()[0]/1
x_l = tap_coords()[1]/1

c_n_list = []
c_a_list = []
c_m_le_list = []
c_m_fourth_list = []
c_l_list = []
c_d_list = []
c_d_w_list = []
c_d_w_ac_list = []
x_cp_c_list = []
u_1_list = []


def C_p_a(alpha):

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

    c_m_l = (trapezoid(c_p_u, x_u) - trapezoid(c_p_l, x_l))/100**2

    return c_m_l

def C_p_t(alpha):

    c_p_t_list = []

    general_alpha = general[general['Alpha'] == alpha]

    p_stat = general_alpha.iloc[0]['p_stat']
    q_inf = general_alpha.iloc[0]['q_inf']

    p_t = total_wake(alpha)[0]
    p_s = static_wake(alpha)[0]

    x_t = total_wake(alpha)[1]
    x_s = static_wake(alpha)[1]

    p_t_func = interp1d(x_t, p_t, kind='linear', fill_value="extrapolate")
    p_s_func = interp1d(x_s, p_s, kind='linear', fill_value="extrapolate")

    for i in range(0,len(x_s)):
        c_p_t_list.append((p_t_func(x_s[i]) - p_s_func(x_s[i]))/q_inf)

    return c_p_t_list, x_s

def C_p_t_func(alpha):

    c_p_t_list = []

    general_alpha = general[general['Alpha'] == alpha]

    p_stat = general_alpha.iloc[0]['p_stat']
    q_inf = general_alpha.iloc[0]['q_inf']

    p_t = total_wake(alpha)[0]
    p_s = static_wake(alpha)[0]

    x_t = total_wake(alpha)[1]
    x_s = static_wake(alpha)[1]

    p_t_func = interp1d(x_t, p_t, kind='linear', fill_value="extrapolate")
    p_s_func = interp1d(x_s, p_s, kind='linear', fill_value="extrapolate")

    x_t_accurate = np.linspace(x_t[0], x_t[len(x_t)-1], 200)

    for i in range(0,len(x_t_accurate)):
        c_p_t_list.append((p_t_func(x_t_accurate[i]) - p_stat)/q_inf)

    return c_p_t_list, x_t_accurate

def U_1(alpha):

    c_p_t_list = []

    general_alpha = general[general['Alpha'] == alpha]

    p_stat = general_alpha.iloc[0]['p_stat']
    rho = general_alpha.iloc[0]['rho']
    q_inf = general_alpha.iloc[0]['q_inf']

    p_t = total_wake(alpha)[0]
    p_s = static_wake(alpha)[0]

    x_t = total_wake(alpha)[1]
    x_s = static_wake(alpha)[1]

    p_t_func = interp1d(x_t, p_t, kind='linear', fill_value="extrapolate")
    p_s_func = interp1d(x_s, p_s, kind='linear', fill_value="extrapolate")


    for i in range(0,len(x_t)):
        u_1_list.append((2*(p_t_func(x_t[i]) - p_stat)/rho)**(1/2))

    return u_1_list, x_t

#Surface pressure data
for i in range(len(general['Alpha'])):

    c_p_u = C_p_a(general['Alpha'][i])[0]
    c_p_l = C_p_a(general['Alpha'][i])[1]
    

    c_p_t_u = np.linspace(0,0, len(c_p_u))
    c_p_t_l = np.linspace(0,0, len(c_p_l))

    for i in range(0,len(c_p_u)):
        #print(x_u.iloc[i])
        c_p_t_u[i] = c_p_u.iloc[i]*upper_surface_angle(x_u.iloc[i])

    for i in range(0,len(c_p_l)):
        c_p_t_l[i] = c_p_l.iloc[i]*lower_surface_angle(x_l.iloc[i])


    c_n = (trapezoid(c_p_l,x_l) - trapezoid(c_p_u,x_u))/100 
    c_a = (trapezoid(c_p_t_u,x_u) - trapezoid(c_p_t_l,x_l))

    c_m_le = C_m_le(c_p_u, c_p_l)
    c_m_fourth = c_m_le + 0.25 * c_n

    aoa_rad = math.radians(general['Alpha'][i])

    c_l = c_n*(math.cos(aoa_rad) - (math.sin(aoa_rad)**2)/math.cos(aoa_rad))
    c_d = - c_a*math.cos(aoa_rad) + c_n*math.sin(aoa_rad)

    x_cp_c = -c_m_le/c_n

    c_n_list.append(c_n)
    c_a_list.append(-c_a)
    c_m_le_list.append(c_m_le)
    c_m_fourth_list.append(c_m_fourth)
    c_l_list.append(c_l)
    c_d_list.append(c_d)
    x_cp_c_list.append(x_cp_c*160)

#Wake rake data
for i in range(len(general['Alpha'])):
    c_p_t, x_s = C_p_t(general['Alpha'][i])

    c_p_t_ac, x_s_ac = C_p_t_func(general['Alpha'][i])

    #if i % 8 == 0:
        #plt.plot(x_s_ac, c_p_t_ac, label= f"c_p_t at {general['Alpha'][i]} deg")

    
    sqrt_c_p_t = np.linspace(0,0,len(c_p_t))

    sqrt_c_p_t_ac = np.linspace(0,0,len(c_p_t_ac))


    for i in range(0,len(c_p_t)):
        sqrt_c_p_t[i] = math.sqrt(c_p_t[i])

    for i in range(0,len(c_p_t_ac)):
        sqrt_c_p_t_ac[i] = math.sqrt(c_p_t_ac[i])


    c_d = -trapezoid(((sqrt_c_p_t)*(1-sqrt_c_p_t)), x_s)*2/219

    c_d_ac = trapezoid(((sqrt_c_p_t_ac)*(1-sqrt_c_p_t_ac)), x_s_ac-132/2-43.5)*2/219

    c_d_w_list.append(c_d)

    c_d_w_ac_list.append(c_d_ac)

for i in range(0, len(c_l_list)):

    aoa_rad = math.radians(general['Alpha'][i])
    
    c_l_list[i] -= c_d_w_ac_list[i] * math.tan(aoa_rad)


desired_alpha = 8

c_p_u = C_p_a(desired_alpha)[0]
c_p_l = C_p_a(desired_alpha)[1]

u_1, x_t = U_1(desired_alpha)

#plt.plot(x_t, u_1, marker='o', label="U1 Velocity Profile")

# #Plotting Cp chordwise for a desired alpha
# plt.plot(x_u, c_p_u, marker='o', label="Upper Surface")
# plt.plot(x_l, c_p_l, marker='o', label="Lower Surface")
# plt.xlabel('Position (x [mm])')
# plt.ylabel('Cp, Pressure Coefficient')
# plt.title(f'Pressure Coefficient Distribution for Angle of Attack = {desired_alpha}°')

# plt.gca().invert_yaxis()  # Get the current axes and invert the y-axis
# plt.grid(True)
# plt.legend()
# plt.show()

import csv

# Two separate lists (one for names and one for ages)
aoa = c_d_w_ac_list
coeff1 = c_l_list   


        # Combine the two lists into rows (pairs of name and age)
data = zip(aoa, coeff1)

        # Specify the name of the CSV file
filename = f"cd-cl.csv"

        # Open the CSV file in write mode
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
            
            # Write the header (optional)
    writer.writerow(["cd", "cl"])
            
            # Write each row (name, age) to the CSV file
    writer.writerows(data)

print(f"CSV file '{filename}' has been created.")


# Plotting Coeffs over alpha
#plt.plot(general['Alpha'], c_n_list, label="C_n")
#plt.plot(general['Alpha'], c_a_list, label="c_a")
# plt.plot(general['Alpha'], c_m_le_list, label="C_m_le")
# plt.plot(general['Alpha'], c_m_fourth_list, marker='.', label="C_m_c/4", color='black')
#plt.plot(general['Alpha'], c_l_list, marker='.', label="C_l", color='red')
#plt.plot(general['Alpha'], c_d_list, label="C_d(fake)")
# plt.plot(general['Alpha'], c_d_w_ac_list, marker='.', label="C_d", color='orange')
plt.plot(c_d_w_ac_list, c_l_list,marker='.', label="Cl-Cd)", color='orange')
#plt.plot(general['Alpha'], x_cp_c_list, marker='o', label="x_cp")
plt.ylabel('C_l, Coefficient of lift [-]')
plt.xlabel('C_d, Coefficient of lift[-]')
plt.legend()
plt.grid(True)
#plt.show()