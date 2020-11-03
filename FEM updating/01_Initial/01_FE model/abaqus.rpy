# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2017 replay file
# Internal Version: 2016_09_27-23.54.59 126836
# Run by bjorntsv on Tue Nov 03 11:18:41 2020
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.45752, 1.45833), width=214.547, 
    height=144.667)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('ss_beam.py', __main__.__dict__)
#: The model "Model-1" has been created.
#: The model database has been saved to "C:\Users\bjorntsv\Dropbox\02_PhD\06_Journal papers\02_JP2\06_FEM updating\01_Numerical study\07_Github\FEM updating\01_Initial\01_FE model\ss_beam.cae".
#: Model: C:/Users/bjorntsv/Dropbox/02_PhD/06_Journal papers/02_JP2/06_FEM updating/01_Numerical study/07_Github/FEM updating/01_Initial/01_FE model/Analysis1.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             2
#: Number of Element Sets:       3
#: Number of Node Sets:          2
#: Number of Steps:              1
print 'RT script done'
#: RT script done
