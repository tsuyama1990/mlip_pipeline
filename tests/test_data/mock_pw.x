#!/bin/bash
# Mock pw.x script for integration testing

# Find the input and output file paths from the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -in) input_file="$2"; shift ;;
        *) ;;
    esac
    shift
done

# Default output file if not specified (pw.x behavior)
output_file="${input_file%.in}.out"

# The mock output content
cat << 'EOF' > "$output_file"
     Program PWSCF v.7.2 starts on 22Apr2024 at 19:48:38

     This program is part of the open-source Quantum ESPRESSO suite
     for quantum simulation of materials; please cite
     "P. Giannozzi et al., J. Phys.: Condens. Matter 21 395502 (2009);
      "P. Giannozzi et al., J. Phys.: Condens. Matter 29 465901 (2017);
      URL http://www.quantum-espresso.org",
      in publications based on this work.

     Parallel version (MPI), running on     1 processor

number of atoms/cell      = 2
number of atomic types    = 1

celldm(1)= 1.8897261246 celldm(2)= 1.0 celldm(3)= 1.0 celldm(4)= 0.0 celldm(5)= 0.0 celldm(6)= 0.0
crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

site n.     atom                  positions (alat units)
   1           H          tau(   1) = (   0.00000000   0.00000000   0.00000000  )
   2           H          tau(   2) = (   0.00000000   0.00000000   0.74000000  )

!    total energy              =     -16.42531639 Ry
     Harris-Foulkes estimate   =     -16.42531639 Ry
     est. exchange-corr. energy=      -5.32059333 Ry

     The total energy is the sum of the following terms:
     one-electron contribution =       9.11184341 Ry
     hartree contribution      =       1.41312782 Ry
     xc contribution           =      -5.32059333 Ry
     ewald contribution        =     -21.62969429 Ry

     Forces acting on atoms (Ry/au):

     atom    1 type  H  force =   -0.00000135   -0.00000000    0.00000000
     atom    2 type  H  force =    0.00000135    0.00000000    0.00000000

     Total force =      0.000003     Total SCF correction =      0.000000

     entering subroutine stress ...
     total   stress  (Ry/bohr**3)            (kbar)     P=      -0.01
  -0.00000001   0.00000000   0.00000000        -0.00      0.00      0.00
   0.00000000  -0.00000001   0.00000000         0.00     -0.00      0.00
   0.00000000   0.00000000  -0.00000001         0.00      0.00     -0.00


     A final scf cycle is performed
     without symmetrization to find the representation of density
     and potentials in the original basis

!    total energy              =     -16.42531639 Ry

     JOB DONE.
EOF
