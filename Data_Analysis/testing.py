import pandas as pd

# Example DataFrames
df1 = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

df2 = pd.DataFrame({
    'X': [9, 10, 11, 12],
    'Y': [13, 14, 15, 16]
})

# Adding a column from df1 to df2
df2['new_column'] = df1['A']

# Now df2 has the new column from df1
print(df2)