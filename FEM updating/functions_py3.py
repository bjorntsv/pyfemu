# -*- coding: utf-8 -*-
"""
functions_py3.py

Assembly of relevant functions adapted to Python 3.
"""


import numpy as np
from collections import OrderedDict


def get_MAC_MMI(freqs_est, modes_est, freqs_num, modes_num, filtering=False,
                tol_upper=0.025, tol_lower=-0.025):
    """MAC and Mode Match Index (MMI) based on estimated and numerical results.

    This function establishes MAC and MMI and returns the information in
    matrices. The function also performs filtering of (local) numerical mode
    shapes based on a tolerance.

    Note that the format of the information contained in the estimated and
    numerical results for frequencies and mode shapes must be similar.

    Parameters
    ----------
    freqs_est : ndarray
        Estimated frequencies.
    modes_est : ndarray
        Estimated mode shapes.
    freqs_num : ndarray
        Numerically predicted frequencies.
    modes_num : ndarray
        Numerically predicted mode shapes.
    filtering : bool, optional
        When True, numerical mode shapes are filtered based on an upper and
        lower tolerance value. Default is False.
    tol_upper : float, optional
        Maximum upper tolerance value. Defaults to 0.025.
    tol_lower : float, optional
        Minimum lower tolerance value. Defaults to -0.025.

    Returns
    -------
    MAC_matrix : ndarray
        Filtered MAC matrix.
    MMI_matrix : ndarray
        Filtered MMI matrix.
    abaqus_filter_id_list : list
        List of id's of the filtered numerical mode shapes.
    results_MAC : dict
        Dictionary with the ten best MAC values (sorted) for each estimated
        mode shape evaluated.
    results_MMI : dict
        Dictionary with the ten best MMI values (sorted) for each estimated
        mode shape evaluated.
    """

    # ---------------- #
    # MAC AND MMI
    # ---------------- #

    # General parameters
    rows_est, cols_est = np.shape(modes_est)
    rows_num, cols_num = np.shape(modes_num)

    if filtering:

        # Abaqus parameters
        abaqus_filter_id_list = []
        tol_max = tol_upper
        tol_min = tol_lower

    # MAC parameters
    MAC_matrix = np.zeros((cols_num, cols_est), dtype=float)

    # Mode match index parameters
    gamma = 0.5
    MMI_matrix = np.zeros((cols_num, cols_est), dtype=float)

    # Loop through estimated mode shapes
    for i in range(cols_est):

        # Estimated mode shape vector and corresponding frequency considered
        phi_e = modes_est[:, i]
        f_e = freqs_est[i, 1]

        # Loop through numerical mode shapes
        for j in range(cols_num):

            if filtering:

                # Filter out local numerical modes
                mode_max = np.amax(modes_num[:, j])
                mode_min = np.amin(modes_num[:, j])

                # Check for local Abaqus mode shapes
                if mode_max < tol_max and mode_min > tol_min:

                    # Append to list (only append one time, i == 0)
                    if i == 0:
                        abaqus_id = j+1
                        abaqus_filter_id_list.append(abaqus_id)

                    # Continue to next numerical mode shape
                    continue

            # Numerical mode shaper vector and corresponding frequency
            # considered
            phi_n = modes_num[:, j]
            f_n = freqs_num[j, 1]

            # MAC value
            MAC_value = (np.abs(phi_n@phi_e))**2 / ((phi_n@phi_n)*(phi_e@phi_e))
            MAC = round(MAC_value, 5)

            # MMI value
            MMI = (1 - gamma)*MAC - gamma*(np.abs(f_e - f_n) / f_e)

            # Append
            MAC_matrix[j, i] = MAC
            MMI_matrix[j, i] = MMI

    # ---------------- #
    # SYSTEM ID
    # ---------------- #

    # Information matrix
    SI = np.zeros((cols_num, 4), dtype=float)
    SI[:, 2] = freqs_num[:, 1]
    SI[:, 3] = np.arange(1, cols_num+1, 1)

    # Dictionary with all results sorted by mode matching results and
    # MAC results
    results_MMI = OrderedDict([])
    results_MAC = OrderedDict([])

    # For each mode estimated by OMA, finding the maximum mode matching index
    # (MMI) and the corresponding MAC, frequency and Abaqus ID.
    for k in range(cols_est):

        # Estimated mode shape considered
        mode_est = 'Mode ' + str('{:02}'.format(k+1))

        # MMI
        SI[:, 0] = MMI_matrix[:, k]

        # MAC
        SI[:, 1] = MAC_matrix[:, k]

        # Sort by MMI
        idx_MMI = np.argsort(SI[:, 0], axis=0)
        SI_sort_MMI_asc = SI[idx_MMI]
        SI_sort_MMI_dsc = SI[idx_MMI][::-1]

        # Sort by MAC
        idx_MAC = np.argsort(SI[:, 1], axis=0)
        SI_sort_MAC_asc = SI[idx_MAC]
        SI_sort_MAC_dsc = SI[idx_MAC][::-1]

        # Append the 10 best results
        results_MMI.update({mode_est: SI_sort_MMI_dsc[:10]})
        results_MAC.update({mode_est: SI_sort_MAC_dsc[:10]})

    # ---------------- #
    # RETURN
    # ---------------- #

    # Return
    if filtering:

        return(MAC_matrix, MMI_matrix, abaqus_filter_id_list,
               results_MAC, results_MMI)

    else:

        return(MAC_matrix, MMI_matrix, results_MAC, results_MMI)
