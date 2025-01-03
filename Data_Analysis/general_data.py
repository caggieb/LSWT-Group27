import pandas as pd
import matplotlib.pyplot as plt


pressure_df = pd.read_csv("raw_2D.csv", skiprows=0)
pressure_df = pressure_df.astype(float)



dP_b = pressure_df['Delta_Pb']
P_t =  pressure_df['P097']
P_b =  pressure_df['P_bar']
rho = pressure_df['rho']

q_inf = 0.211804 + 1.928442*dP_b + 1.879374 * dP_b * 10**(-4) 
p_stat = P_t - q_inf

print(p_stat)

Alpha = pressure_df['Alpha']

result = pd.concat([Alpha, q_inf, p_stat, P_b, rho], axis=1)

result = result.rename(columns={'Delta_Pb':'q_inf', 0:'p_stat'})

result.to_csv('general_data.csv', index=False)


print(pressure_df)

print(result)