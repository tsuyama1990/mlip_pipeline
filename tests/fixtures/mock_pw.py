import sys

# Minimal QE Output
output = """
     Program PWSCF v.6.4.1 starts on  9Sep2019 at 16:36:12

     Parallel version (MPI), running on     1 processors

     bravais-lattice index     =            0
     lattice parameter (alat)  =      18.8973  a.u.
     unit-cell volume          =    6748.3344 (a.u.)^3
     number of atoms/cell      =            2
     number of atomic types    =            1

     celldm(1)=  18.897261  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     site n.     atom                  positions (alat units)
         1           H   tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           H   tau(   2) = (   0.0000000   0.0000000   0.0750000  )

!    total energy              =     -13.60567890 Ry
     Harris-Foulkes estimate   =     -13.60567890 Ry
     estimated scf accuracy    <          1.8E-10 Ry

     convergence has been achieved in   6 iterations

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00100000    0.00200000    0.00300000
     atom    2 type  1   force =    -0.00100000   -0.00200000   -0.00300000

     Total force =     0.000000     Total SCF correction =     0.000000


          total   stress  (Ry/bohr**3)                   (kbar)     P=   -0.00
   -0.00002369   0.00000000   0.00000000        -3.48      0.00      0.00
    0.00000000  -0.00002369   0.00000000         0.00     -3.48      0.00
    0.00000000   0.00000000  -0.00002369         0.00      0.00     -3.48

     JOB DONE.
"""

# Print output to stdout
print(output)
