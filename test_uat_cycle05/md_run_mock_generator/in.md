units metal
atom_style atomic
boundary p p p
read_data structure.data
mass * 1.0
pair_style hybrid/overlay pace zbl 0.5 1.2
pair_coeff * * pace potential.yace He
pair_coeff 1 1 zbl 2 2
neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes
compute pace all pace potential.yace He
compute max_gamma all reduce max c_pace[1]
variable check_gamma equal c_max_gamma>5.0
fix halt_check all halt 1 v_check_gamma != 0 error 100
timestep 0.001
velocity all create 300.0 12345 mom yes rot yes dist gaussian
fix 1 all nvt temp 300.0 300.0 $(100.0*dt)
thermo 10
thermo_style custom step temp press etotal c_max_gamma
dump 1 all custom 100 traj.dump id type x y z c_pace[1]
dump_modify 1 sort id
run 100