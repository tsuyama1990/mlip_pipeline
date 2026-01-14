#!/usr/bin/env python
# coding: utf-8

# # CYCLE01 User Acceptance Test: Silicon Equation of State

# This notebook verifies the core functionality of CYCLE01 by generating a dataset for the equation of state (EOS) of silicon and running DFT calculations on it.

# In[ ]:


import yaml
from pathlib import Path

# 1. Configuration
user_config = {
    'project_name': 'si_eos',
    'target_system': {
        'elements': ['Si'],
        'composition': {'Si': 1.0},
        'crystal_structure': 'diamond',
    },
    'generation_config': {'generation_type': 'eos_strain'},
}

config_path = Path('si_eos_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(user_config, f)


# 2. Execution

# In[ ]:


import subprocess
subprocess.run(['python', '-m', 'mlip_autopipec', str(config_path)])


# 3. Verification and Visualization

# In[ ]:


from ase.db import connect
import matplotlib.pyplot as plt

db = connect('si_eos.db')
volumes = []
energies = []

for row in db.select():
    volumes.append(row.volume)
    energies.append(row.energy)

plt.plot(volumes, energies, 'o-')
plt.xlabel('Volume (Å³)')
plt.ylabel('Energy (eV)')
plt.title('Silicon Equation of State')
plt.savefig('si_eos.png')
plt.show()
