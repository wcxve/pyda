"""
    Test for Jacoby.py bases on
    http://jean-pierre.moreau.pagesperso-orange.fr/Fortran/tujacobi_f90.txt

    Output:
    NROT= 25
    [ 7.  1.  3.  5.]
    [[ 0.5  0.5 -0.5 -0.5]
     [-0.5  0.5 -0.5  0.5]
     [-0.5  0.5  0.5 -0.5]
     [ 0.5  0.5  0.5  0.5]]
"""
import numpy as np
import jacoby

m=[ [ 4.0,-2.0,-1.0, 0.0],
    [-2.0, 4.0, 0.0,-1.0],
    [-1.0, 0.0, 4.0,-2.0],
    [ 0.0,-1.0,-2.0, 4.0]]

A = np.array(m)
N = 4
D = np.zeros(4)
V = np.zeros([4,4])
NROT=jacoby.Jacobi(A,N,D,V)

print('NROT=',NROT)
print(D)
print(V)
