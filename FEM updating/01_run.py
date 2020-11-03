# -- coding utf-8 --

"""
01_run.py

File for performing FEM updating. Results from the FE model updating are
stored in the file info.pkl.
"""

import os
import shutil
import pickle
import time
import numpy as np

from scipy.optimize import lsq_linear
from collections import OrderedDict
from functions_py3 import *


# Total time - start
total_time_start = time.time()

# -------------------------------------------------------------------------- #
# ANALYSIS SETTINGS
# -------------------------------------------------------------------------- #

# Number of iterations to perform
iterations = 5

# Measured frequencies to perform FE model updating on
mode_list = ['Mode 01', 'Mode 02', 'Mode 03', 'Mode 04']


# -------------------------------------------------------------------------- #
# INITIAL
# -------------------------------------------------------------------------- #

# Folder structure
folder_1 = '02_Analysis files'
folder_2 = '03_Results/01_Perturbed'
folder_3 = '04_Figures'

folders = [folder_1, folder_2, folder_3]

# Establish folder structure
for folder in folders:

    # Remove existing folders with content
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        time.sleep(1)

    # Create folder
    os.makedirs(folder)

# Import measured frequencies and mode shapes
freqs_est = np.load('01_Initial/02_Measured/ss_beam_0_frequencies_all.npy')
modes_est = np.load('01_Initial/02_Measured/ss_beam_0_modes_all.npy')

# Copy relevant model files to directory
src = ('01_Initial/01_FE model/')

files_to_copy = ['ss_beam.cae', 'ss_beam.jnl']

# Copy
for file in files_to_copy:

    shutil.copy(src + file, file)

# Set work directory
cwd = os.getcwd()
os.chdir(cwd)


# ---------------------------------------------------------------------------#
# ABAQUS MODEL
# ---------------------------------------------------------------------------#

# ------------------------------------ #
# FEM UPDATING PARAMETERS
# ------------------------------------ #

# FEM initial updating parameters
theta1 = 2.415E11
theta2 = 50
theta3 = 25

# Parameter list
theta_0 = np.array([theta1, theta2, theta3])


# ------------------------------------ #
# FEM ANALYSIS
# ------------------------------------ #

# ---------------- #
# EXPORT VARIABLES
# ---------------- #

# Job name
job_name = 'Analysis1'
with open('job_name_list' + '.txt', 'w') as fout:
    fout.write(job_name)

# Parameter list
np.save('parameter_list' + '.npy', theta_0, allow_pickle=True,
        fix_imports=True)

# ---------------- #
# RUN ANALYSIS
# ---------------- #

# Variables
script_name = 'abaqus_ss_beam_upd'

# Run script
print('--------------------------------------------')
print('Initial analysis (1)')
t0 = time.time()

os.system('abaqus cae noGUI=' + script_name)

t1 = time.time()
print('Initial analysis (1) done - ' + str(round(t1 - t0, 3)) + ' sec.')
print('--------------------------------------------')
print()

# Wait 3 sec for the Abaqus analysis to properly close
time.sleep(3)

# ------------------------------------ #
# POST-PROCESS
# ------------------------------------ #

# Import
import_folder = '03_Results/'

freqs_num = np.load(import_folder + job_name + '_frequencies_all.npy')
modes_num = np.load(import_folder + job_name + '_modes_all.npy')

# Sorted MAC and MMI
(MAC_initial_model, _, results_MAC, _) = get_MAC_MMI(freqs_est, modes_est,
                                                     freqs_num, modes_num,
                                                     filtering=False)

# General parameters
rows_est, cols_est = np.shape(modes_est)
rows_num, cols_num = np.shape(modes_num)


# ---------------------------------------------------------------------------#
# FEM UPDATING
# ---------------------------------------------------------------------------#

# ------------------------------------ #
# INITIALIZATION
# ------------------------------------ #

# Variables
q = len(mode_list)
p = len(theta_0)

# Declaration of variables
# Variables saved for each iteration
theta = np.zeros((iterations + 1, p), dtype=float)
lamda = np.zeros((iterations + 1, q), dtype=float)
lamda_all = np.zeros((iterations + 1, len(freqs_est)), dtype=float)
lamda_m = np.zeros(len(mode_list), dtype=float)

G_matrices = OrderedDict([])
MAC_updated_model = OrderedDict([])
results_MAC_updated_model = OrderedDict([])
results_MAC_pt_all_matrices = OrderedDict([])
J_star_list = []

# Start values for the iterations
for idx, mode in enumerate(mode_list):

    # Establish frequencies based on the mode_list string
    string_idx = mode[-2:]

    # Correct for 0 in the string
    if string_idx[0] == 0:

        f_idx = int(mode[-1]) - 1

    else:
        f_idx = int(string_idx) - 1

    # Measured frequencies (to perform FEM updating on)
    lamda_m[idx] = freqs_est[:, 1][f_idx]

    # Numerical frequencies (to perform FEM updating on)
    # Top match based on MAC
    lamda[0, idx] = results_MAC[mode][0, 2]

# All numerical frequencies
for idx, mode in enumerate(results_MAC.keys()):

    lamda_all[0, idx] = results_MAC[mode][0, 2]

# Updating parameters
theta[0, :] = np.copy(theta_0)
d_theta_start = 0.001*theta_0

# Weighting matrix
W = np.linalg.inv(np.diag(lamda_m))**2

# ------------------------------------ #
# OBJECTIVE FUNCTION
# ------------------------------------ #

# Evaluate the objective function for the initial analysis
J_star = np.sum(np.diag(W)*((lamda_m - lamda[0, :])/lamda[0, :])**2)
J_star_list.append(J_star)


# ------------------------------------ #
# ITERATIONS
# ------------------------------------ #

for itr in range(iterations):

    # First iteration step
    if itr == 0:

        # Initialize the parameter incremenet
        d_theta = d_theta_start

    # All other iteration steps
    else:

        # Update the parameter increment
        d_theta = d_theta_upd

    # Variable
    results_MAC_pt_matrices = OrderedDict([])

    # -------------------------- #
    # SENSITIVITY MATRIX
    # -------------------------- #

    # Sensitivity matrix (scaled)
    G = np.zeros((q, p), dtype=float)

    # Loop through the parameters (columns)
    for ii in range(p):

        # -------------------------- #
        # PERTURBATION - THETA
        # -------------------------- #

        # Perturbed parameter vector
        theta_pt = np.copy(theta[itr, :])
        theta_pt[ii] = theta_pt[ii] + d_theta[ii]

        # -------------------------- #
        # PERTURBATION - LAMDA
        # -------------------------- #

        # -------------------------- #
        # FE ANALYSIS
        # -------------------------- #

        # ---------------- #
        # EXPORT VARIABLES
        # ---------------- #

        # Job name
        job_name = 'Analysis' + str(itr+1) + '_pt' + str(ii+1)
        with open('job_name_list' + '.txt', 'a') as fout:
            fout.write('\n' + job_name)

        # Parameter list
        np.save('parameter_list_pt' + '.npy', theta_pt, allow_pickle=True,
                fix_imports=True)

        # ---------------- #
        # RUN ANALYSIS
        # ---------------- #

        # Variables
        script_name = 'abaqus_ss_beam_upd_pt'

        # Run script
        print('Perturbation analysis ' + str(ii+1))
        t0_pt = time.time()

        os.system('abaqus cae noGUI=' + script_name)

        t1_pt = time.time()
        print('Perturbation analysis ' + str(ii+1) + ' done - ' +
              str(round(t1_pt - t0_pt, 3)) + ' sec.')
        print('--------------------------------------------')
        print()

        # Wait 3 sec for the Abaqus analysis to properly close
        time.sleep(3)

        # -------------------------- #
        # POST-PROCESS
        # -------------------------- #

        # Import
        import_folder = '03_Results/01_Perturbed/'

        freqs_num_pt = np.load(import_folder + job_name +
                               '_frequencies_all.npy')

        modes_num_pt = np.load(import_folder + job_name + '_modes_all.npy')

        # Sorted MAC and MMI
        (_, _, results_MAC_pt, _) = get_MAC_MMI(freqs_est, modes_est,
                                                freqs_num_pt, modes_num_pt,
                                                filtering=False)

        # Append perturbed results for validation
        results_MAC_pt_matrices.update({'results_MAC_pt' + str(ii+1) : 
                                        results_MAC_pt})

        # Perturbed numerical frequencies
        lamda_pt = np.zeros(q, dtype=float)
        lamda_pt_all = np.zeros(cols_est, dtype=float)

        for idx, mode in enumerate(mode_list):

            # Establish frequencies based on the mode_list string
            string_idx = mode[-2:]

            # Correct for 0 in the string
            if string_idx[0] == 0:

                f_idx = int(mode[-1]) - 1

            else:
                f_idx = int(string_idx) - 1

            # Perturbed numerical frequencies
            # Top match based on MAC
            lamda_pt[idx] = results_MAC_pt[mode][0, 2]

        # All perturbed numerical frequencies
        for idx, mode in enumerate(results_MAC_pt.keys()):

            lamda_pt_all[idx] = results_MAC_pt[mode][0, 2]

        # Sensitivities
        # Loop through the frequencies (rows)
        for jj in range(q):

            # Numerator
            nmr = (lamda_pt[jj] - lamda[itr, jj]) / lamda[0, jj]

            # Denominator
            dnm = (theta_pt[ii] - theta[itr, ii]) / theta[0, ii]

            # Update sensitivity matrix
            G[jj, ii] = nmr/dnm

    # Append
    G_matrices.update({'G' + str(itr+1): G})

    # Append
    results_MAC_pt_all_matrices.update({'results_MAC_pt_matrices_' +
                                        str(itr+1): results_MAC_pt_matrices})

    # -------------------------- #
    # OPTIMIZATION
    # -------------------------- #

    # Residual, r, scaled
    r_s = (lamda_m - lamda[itr, :]) / lamda[0, :]

    # Global parameter bounds
    lower_allowable_theta = np.array([-np.inf, -np.inf, -np.inf])
    upper_allowable_theta = np.array([np.inf, np.inf, np.inf])

    # Local parameter bounds
    lb_local = np.array([0.5, 0.5, 0.5])
    ub_local = np.array([1.5, 1.5, 1.5])

    # Local parameter bounds, unscaled
    theta_min = lb_local*theta[itr, :]
    theta_max = ub_local*theta[itr, :]

    # Check for exceedance of global bounds
    for index in range(p):

        if theta_min[index] <= lower_allowable_theta[index]:
            theta_min[index] = np.copy(lower_allowable_theta[index])

        if theta_max[index] >= upper_allowable_theta[index]:
            theta_max[index] = np.copy(upper_allowable_theta[index])

    # Upper and lower parameter bounds
    d_theta_min = theta_min - theta[itr, :]
    d_theta_max = theta_max - theta[itr, :]

    # Upper and lower parameter bounds, scaled
    d_theta_min_s = d_theta_min / theta[0, :]
    d_theta_max_s = d_theta_max / theta[0, :]

    # Minimization variables
    A = -np.sqrt(W) @ G
    b = -np.sqrt(W) @ r_s
    lb = np.copy(d_theta_min_s)
    ub = np.copy(d_theta_max_s)

    # Minimization algorithm
    res = lsq_linear(A, b, bounds=(lb, ub), method='trf',
                     lsq_solver=None, lsmr_tol='auto', verbose=1)

    # -------------------------- #
    # UPDATE
    # -------------------------- #

    # Parameter incremental, d_theta, scaled
    d_theta_s_upd = np.copy(res.x)

    # Update d_theta_upd (unscaled)
    d_theta_upd = d_theta_s_upd*theta[0, :]

    # Update theta
    theta[itr+1, :] = theta[itr, :] + d_theta_upd

    # ------------------------------------ #
    # FEM ANALYSIS
    # ------------------------------------ #

    # ---------------- #
    # EXPORT VARIABLES
    # ---------------- #

    # Job name
    job_name = 'Analysis' + str(itr+2)
    with open('job_name_list' + '.txt', 'a') as fout:
        fout.write('\n' + job_name)

    # Parameter list
    np.save('parameter_list_upd' + '.npy', theta[itr+1, :], allow_pickle=True,
            fix_imports=True)

    # ---------------- #
    # RUN ANALYSIS
    # ---------------- #

    # Variables
    script_name = 'abaqus_ss_beam_upd'

    # Run script
    print()
    print('--------------------------------------------')
    print('Updated analysis (' + str(itr+2) + ')')
    t0 = time.time()

    os.system('abaqus cae noGUI=' + script_name)

    t1 = time.time()
    print('Updated analysis (' + str(itr+2) + ') done - ' +
          str(round(t1 - t0, 3)) + ' sec.')
    print('--------------------------------------------')
    print()

    # Wait 3 sec for the Abaqus analysis to properly close
    time.sleep(3)

    # ------------------------------------ #
    # POST-PROCESS
    # ------------------------------------ #

    # Import
    import_folder = '03_Results/'

    freqs_num_upd = np.load(import_folder + job_name + '_frequencies_all.npy')
    modes_num_upd = np.load(import_folder + job_name + '_modes_all.npy')

    # Sorted MAC and MMI
    (MAC_updated, _, results_MAC_upd, _) = get_MAC_MMI(freqs_est,
                                                       modes_est,
                                                       freqs_num_upd,
                                                       modes_num_upd,
                                                       filtering=False)

    # Updated frequencies
    lamda_upd = np.zeros(q, dtype=float)
    lamda_upd_all = np.zeros(cols_est, dtype=float)

    for idx, mode in enumerate(mode_list):

        # Establish frequencies based on the mode_list string
        string_idx = mode[-2:]

        # Correct for 0 in the string
        if string_idx[0] == 0:

            f_idx = int(mode[-1]) - 1

        else:
            f_idx = int(string_idx) - 1

        # Updated numerical frequencies
        # Top match based on MAC
        lamda_upd[idx] = results_MAC_upd[mode][0, 2]

    # All updated numerical frequencies
    for idx, mode in enumerate(results_MAC_upd.keys()):

        lamda_upd_all[idx] = results_MAC_upd[mode][0, 2]

    # Update lamda
    lamda[itr+1, :] = np.copy(lamda_upd)
    lamda_all[itr+1, :] = np.copy(lamda_upd_all)

    # Append updated MAC results
    MAC_updated_model.update({'MAC_updated' + str(itr+1): MAC_updated})
    results_MAC_updated_model.update({'results_MAC_upd' + str(itr+1):
                                      results_MAC_upd})

    # -------------------------- #
    # OBJECTIVE FUNCTION
    # -------------------------- #

    # Objective function for evaluation
    J_star = np.sum(np.diag(W)*((lamda_m - lamda[itr+1, :])/lamda[0, :])**2)
    J_star_list.append(J_star)


# Wait 5 sec for the Abaqus analysis to properly close
print('Sleeping 5 sec for Abaqus to properly close.')
time.sleep(5)


# -------------------------------------------------------------------------- #
# MOVE FILES
# -------------------------------------------------------------------------- #

# Move analysis files to folder
dst = '02_Analysis files/'
src_file_list = os.listdir()

# Analysis files
for src_file in src_file_list:

    # Move Abaqus analysis files
    if src_file.startswith('Analysis') or src_file.startswith('abaqus.rpy'):

        shutil.move(src_file, dst)

    # Move Abaqus temporary files
    elif src_file.startswith('temp'):

        shutil.move(src_file, dst)

    # Move other files
    elif src_file.endswith('.npy'):

        shutil.move(src_file, dst)


# -------------------------------------------------------------------------- #
# EXPORT
# -------------------------------------------------------------------------- #

# Export relevant variables to postprocessing
info = {'freqs_est': freqs_est,
        'modes_est': modes_est,
        'freqs_num': freqs_num,
        'modes_num': modes_num,
        'theta': theta,
        'MAC_initial': MAC_initial_model,
        'MAC_updated': MAC_updated_model,
        'lamda_m': lamda_m,
        'lamda': lamda,
        'lamda_all': lamda_all,
        'G_matrices': G_matrices,
        'W': W,
        'J_star_list': J_star_list}

with open('info.pkl', 'wb') as file:
    pickle.dump(info, file)


# -------------------------------------------------------------------------- #
# END OF ANALYSIS
# -------------------------------------------------------------------------- #

# Total time - end
total_time_stop = time.time()
print()
print('Total analysis time is ' +
      str(round((total_time_stop - total_time_start), 1)) + ' sec.')
