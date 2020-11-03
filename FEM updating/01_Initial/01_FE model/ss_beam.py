# -- coding utf-8 --
"""
ss_beam.py

Simply supported beam model.
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


# -------------------------------------------------------------------------- #
# INITIAL
# -------------------------------------------------------------------------- #

# Variables
E = 2.415E11
mass1 = 50
mass2 = 25

# Number of modes to evaluate
n_modes = 6

# ---------------------------------------------------------------------------#
# ABAQUS MODEL
# ---------------------------------------------------------------------------#


# ------------------------------------ #
# MODEL
# ------------------------------------ #

# Model
beamModel = mdb.Model(name='Model-1')

# JournalOptions object (see 47.6 in Abaqus Scripting Reference Guide)
session.journalOptions.setValues(replayGeometry=COORDINATE,
                                 recoverGeometry=COORDINATE)

# Viewport (to display the model and results)
vp1 = session.viewports['Viewport: 1']


# ------------------------------------ #
# MATERIAL
# ------------------------------------ #

# Material
material1 = beamModel.Material(name='Steel')

material1.Density(table=((7850.0, ), ))
material1.Elastic(table=((E, 0.3), ))


# ------------------------------------ #
# SECTION
# ------------------------------------ #

# Pofile
profile1 = mdb.models['Model-1'].RectangularProfile(a=0.05, b=0.1,
                                                    name='R_100_50')

# Beam section
section1 = beamModel.BeamSection(consistentMassMatrix=False,
                                integration=DURING_ANALYSIS,
                                material='Steel', name='Beam_R_100_50',
                                poissonRatio=0.3, profile='R_100_50',
                                temperatureVar=LINEAR)


# ------------------------------------ #
# PART
# ------------------------------------ #

# SKETCH
sketch1 = beamModel.ConstrainedSketch(name='beamSketch', sheetSize=20.0)

point_01 = point=(0.0, 0.0)
point_02 = point=(0.8, 0.0)
point_03 = point=(1.6, 0.0)
point_04 = point=(2.4, 0.0)
point_05 = point=(3.2, 0.0)
point_06 = point=(4.0, 0.0)
point_07 = point=(4.8, 0.0)
point_08 = point=(5.6, 0.0)
point_09 = point=(6.4, 0.0)
point_10 = point=(7.2, 0.0)
point_11 = point=(8.0, 0.0)

sketch1.Line(point1=point_01, point2=point_02)
sketch1.Line(point1=point_02, point2=point_03)
sketch1.Line(point1=point_03, point2=point_04)
sketch1.Line(point1=point_04, point2=point_05)
sketch1.Line(point1=point_05, point2=point_06)
sketch1.Line(point1=point_06, point2=point_07)
sketch1.Line(point1=point_07, point2=point_08)
sketch1.Line(point1=point_08, point2=point_09)
sketch1.Line(point1=point_09, point2=point_10)
sketch1.Line(point1=point_10, point2=point_11)

# CREATE PART
part1 = beamModel.Part(dimensionality=THREE_D, name='Beam',
                       type=DEFORMABLE_BODY)

part1.BaseWire(sketch=sketch1)

# SECTION ASSIGNMENT
part1.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
                        region=Region(edges=part1.edges.findAt(((0.2, 0.0, 0.0), ),
                                                               ((1.0, 0.0, 0.0), ),
                                                               ((1.8, 0.0, 0.0), ),
                                                               ((2.6, 0.0, 0.0), ),
                                                               ((3.4, 0.0, 0.0), ),
                                                               ((4.2, 0.0, 0.0), ),
                                                               ((5.0, 0.0, 0.0), ),
                                                               ((5.8, 0.0, 0.0), ),
                                                               ((6.6, 0.0, 0.0), ),
                                                               ((7.4, 0.0, 0.0), ), )),
                        sectionName='Beam_R_100_50',
                        thicknessAssignment=FROM_SECTION)

# BEAM SECTION ORIENTATION
part1.assignBeamSectionOrientation(method=N1_COSINES, n1=(0.0, 1.0, 0.0),
                                   region=Region(edges=part1.edges.findAt(((0.2, 0.0, 0.0), ),
                                                                          ((1.0, 0.0, 0.0), ),
                                                                          ((1.8, 0.0, 0.0), ),
                                                                          ((2.6, 0.0, 0.0), ),
                                                                          ((3.4, 0.0, 0.0), ),
                                                                          ((4.2, 0.0, 0.0), ),
                                                                          ((5.0, 0.0, 0.0), ),
                                                                          ((5.8, 0.0, 0.0), ),
                                                                          ((6.6, 0.0, 0.0), ),
                                                                          ((7.4, 0.0, 0.0), ), )))

# MESH
regionMesh = (part1.edges.findAt(((0.2, 0.0, 0.0), ), ((1.0, 0.0, 0.0), ),
                                 ((1.8, 0.0, 0.0), ), ((2.6, 0.0, 0.0), ),
                                 ((3.4, 0.0, 0.0), ), ((4.2, 0.0, 0.0), ),
                                 ((5.0, 0.0, 0.0), ), ((5.8, 0.0, 0.0), ),
                                 ((6.6, 0.0, 0.0), ), ((7.4, 0.0, 0.0), ), ), )
elemType = (ElemType(elemCode=B31, elemLibrary=STANDARD), )

# Part 1
part1.setElementType(elemTypes=elemType, regions=regionMesh)
part1.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=0.8)
part1.generateMesh()


# ------------------------------------ #
# ASSEMBLY
# ------------------------------------ #

# Assembly
assembly1 = beamModel.rootAssembly
instance1 = assembly1.Instance(dependent=ON, name='Beam-1', part=part1)

# Assembly set
#assembly1.Set(name='SENSOR_NODES', nodes=instance1.nodes[0:11])
assembly1.Set(name='SENSOR_NODES',
              vertices=instance1.vertices.findAt(((0.0, 0.0, 0.0), ),
                                                 ((0.8, 0.0, 0.0), ),
                                                 ((1.6, 0.0, 0.0), ),
                                                 ((2.4, 0.0, 0.0), ),
                                                 ((3.2, 0.0, 0.0), ),
                                                 ((4.0, 0.0, 0.0), ),
                                                 ((4.8, 0.0, 0.0), ),
                                                 ((5.6, 0.0, 0.0), ),
                                                 ((6.4, 0.0, 0.0), ),
                                                 ((7.2, 0.0, 0.0), ),
                                                 ((8.0, 0.0, 0.0), ), ))


# ------------------------------------ #
# STEP
# ------------------------------------ #

# Step (modal analysis)
beamModel.FrequencyStep(description='Modal analysis', name='Modal', 
                        numEigen=n_modes, previous='Initial')


# ------------------------------------ #
# OUTPUT REQUEST
# ------------------------------------ #

# Field output request
beamModel.fieldOutputRequests['F-Output-1'].setValues(variables=('U', 'RF'))


# ------------------------------------ #
# BOUNDARY CONDITIONS
# ------------------------------------ #

# Boundary conditions
region1 = Region(vertices=instance1.vertices.findAt(((0.0, 0.0, 0.0), ), ))
region2 = Region(vertices=instance1.vertices.findAt(((8.0, 0.0, 0.0), ), ))

# BC1
beamModel.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
                         distributionType=UNIFORM, fieldName='', fixed=OFF,
                         localCsys=None, name='BC1', region=region1, u1=0.0,
                         u2=0.0, u3=0.0, ur1=0.0, ur2=UNSET, ur3=UNSET)

# BC2
beamModel.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
                         distributionType=UNIFORM, fieldName='', fixed=OFF,
                         localCsys=None, name='BC2', region=region2, u1=UNSET,
                         u2=0.0, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET)


# ------------------------------------ #
# LOAD
# ------------------------------------ #

# Point masses
assembly1.engineeringFeatures.PointMassInertia(alpha=0.0, composite=0.0,
                                               mass=mass1, name='M1',
                                               region=Region(vertices=instance1.vertices.findAt(((2.4, 0.0, 0.0), ), )))
assembly1.engineeringFeatures.PointMassInertia(alpha=0.0, composite=0.0,
                                               mass=mass2, name='M2',
                                               region=Region(vertices=instance1.vertices.findAt(((5.6, 0.0, 0.0), ), )))


# ------------------------------------ #
# MESH
# ------------------------------------ #

# Mesh assembly
assembly1.regenerate()


# ---------------------------------------------------------------------------#
# ABAQUS JOB
# ---------------------------------------------------------------------------#

# SAVE
mdb.saveAs(pathName=('ss_beam'))

# JOB
jobName= 'Analysis1'
job1 = mdb.Job(name=jobName, model='Model-1',
               description='Analysis of simply supported beam')

# Submit and wait for the job to complete
job1.submit()
job1.waitForCompletion()


# ---------------------------------------------------------------------------#
# POST-PROCESS
# ---------------------------------------------------------------------------#

# Get data from ODB file
o3 = session.openOdb(name=jobName + '.odb')
odb = session.odbs[jobName + '.odb']

# ------------------------------------ #
# NATURAL FREQUENCIES
# ------------------------------------ #

# Step
step = odb.steps['Modal']

# Region
region = step.historyRegions['Assembly ASSEMBLY']

# History ouputs
modal_freqs = region.historyOutputs['EIGFREQ'].data

# Establish modal data and save
f_matrix = np.zeros((len(modal_freqs), 2), dtype=float)

# Modal data
for i in range(len(modal_freqs)):

    # Arranging the tuple
    mode = int(modal_freqs[i][0])
    mode_freq = modal_freqs[i][1]

    # Save to matrix
    f_matrix[i, 0] = mode
    f_matrix[i, 1] = mode_freq

# Save
np.savetxt('01_Results/' + 'ss_beam_frequencies_all' + '.txt', f_matrix,
           fmt='%.3f', delimiter='\t')
np.save('01_Results/' + 'ss_beam_frequencies_all' + '.npy', f_matrix)


# ------------------------------------ #
# MODE SHAPES
# ------------------------------------ #

# --------------- #
# All modes
# --------------- #

# Note that the resulting file is in a matrix where the columns represents
# the mode shape of the frequency considered.

# Total number of modes investigated
number_of_frames = len(odb.steps['Modal'].frames) - 1

# Node set
node_set = odb.rootAssembly.nodeSets['SENSOR_NODES']
n_sensor_nodes = len(odb.rootAssembly.nodeSets['SENSOR_NODES'].nodes[0])
channels = n_sensor_nodes*3

# Mode shape matrix (33 channels (11 nodes and 3 directions), 10 modes)
phi_matrix = np.zeros((channels, number_of_frames), dtype=float)

# Looping through all modes
for i in range(number_of_frames):

    # Mode
    frame = odb.steps['Modal'].frames[i+1]

    # Field output (displacement)
    displacement = frame.fieldOutputs['U']

    # Set
    node_displacement = displacement.getSubset(region=node_set)

    # Mode shape vector
    phi = np.zeros(channels, dtype=float)

    # Looping through all nodes in the node set
    for idx, node in enumerate(node_displacement.values):

        # Field output data (channel data)
        X = round(node.data[0], 10)
        Y = round(node.data[1], 10)
        Z = round(node.data[2], 10)

        # Save file to column
        # Save to matrix
        phi[(3*idx) + 0] = X
        phi[(3*idx) + 1] = Y
        phi[(3*idx) + 2] = Z

    # Save mode shape vector to mode shape matrix
    phi_matrix[:,i] = phi

# Save to file
np.savetxt('01_Results/' + 'ss_beam_modes_all' + '.txt', phi_matrix,
           fmt='%.10f', delimiter='\t')
np.save('01_Results/' + 'ss_beam_modes_all' + '.npy', phi_matrix)
