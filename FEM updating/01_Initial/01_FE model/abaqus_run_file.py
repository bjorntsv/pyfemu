# -- coding utf-8 --

"""
abaqus_run_file.py

Abaqus run file.
"""

import os


# Set work directory
cwd = os.getcwd()
os.chdir(cwd)

# Variables
scriptName = 'ss_beam'

# Run script
os.system('abaqus cae noGUI=' + scriptName + '.py')
