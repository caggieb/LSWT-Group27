import numpy as np
def quickwrite(a,b,filename="quickwrite.csv", header="Column1, Column2"):
    "a and b must be np arrays"
    wrt = np.vstack((a,b)).T
    np.savetxt(filename, wrt, delimiter=",", header=header, comments="")
    print("arrays quickwritten to:",filename)