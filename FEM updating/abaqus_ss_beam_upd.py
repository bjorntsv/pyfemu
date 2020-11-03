# -- coding utf-8 --
"""
abaqus_ss_beam_upd.py

Analysis of simply supported beam model with updating parameters.
"""

from abaqus import *
from abaqusConstants import *

from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

import numpy as np
from functions_py2 import *


# -------------------------------------------------------------------------- #
# INITIAL
# -------------------------------------------------------------------------- #

# Import
job_name_list = np.loadtxt('job_name_list' + '.txt', dtype=str)

# Initial analysis
if np.size(job_name_list) == 1:
    
    # Job name
    jobName = str(job_name_list)
    
    # Updating parameters
    theta = np.load('parameter_list' + '.npy')

# Updating analyses
else:
    
    # Job name
    jobName = str(job_name_list[-1])

    # Updating parameters
    theta = np.load('parameter_list_upd' + '.npy')


# ---------------------------------------------------------------------------#
# ABAQUS MODEL
# ---------------------------------------------------------------------------#


# ------------------------------------ #
# FEM UPDATING PARAMETERS
# ------------------------------------ #

# FEM initial updating parameters
theta1 = theta[0]
theta2 = theta[1]
theta3 = theta[2]


# ------------------------------------ #
# FEM ANALYSIS
# ------------------------------------ #

# FE model (open model database)
openMdb('ss_beam.cae')

model = mdb.models['Model-1']

# Parameter 1
model.materials['Steel'].elastic.setValues(table=((theta1, 0.3), ))

# Parameter 2 and 3
model.rootAssembly.engineeringFeatures.inertias['M1'].setValues(mass=theta2)
model.rootAssembly.engineeringFeatures.inertias['M2'].setValues(mass=theta3)

# Job
job = mdb.Job(name=jobName, model='Model-1',
              description='Analysis of simply supported beam')

# Submit and wait for the job to complete
job.submit()
job.waitForCompletion()


# ------------------------------------ #
# FEM POSTPROCESSING
# ------------------------------------ #

# Get data from ODB file
o3 = session.openOdb(name=jobName + '.odb')
odb = session.odbs[jobName + '.odb']

# Frequencies
freqs_num = get_frequencies(odb, save=True, name=jobName, folder='03_Results/')

# Mode shapes
modes_num = get_modeshapes(odb, node_set_name='SENSOR_NODES', save=True,
                           name=jobName, folder='03_Results/')
