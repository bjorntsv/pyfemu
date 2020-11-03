# -*- coding: utf-8 -*-
"""
functions_py2.py

Assembly of relevant functions adapted to Python 2.
"""


import numpy as np


def get_frequencies(odb, save=False, name='test', folder='test'):
    """Get frequencies from the Abaqus model.

    Parameters
    ----------
    odb : Abaqus object
        Abaqus object for the ODB file.
    save : bool, optional
        When True, the natural frequencies are saved to .txt and .npy files.
    name : str, optional
        Name to be included of the saved files.
    folder : str, optional
        Name of the folder to include the saved files in.

    Returns
    -------
    f_matrix : ndarray
        Matrix with mode number and corresponding frequency (Hz) [mode, freq]
    """

    # Region
    region = odb.steps['Modal'].historyRegions['Assembly ASSEMBLY']

    # History outputs
    modal_freqs = region.historyOutputs['EIGFREQ'].data

    # Establish modal data
    f_matrix = np.zeros((len(modal_freqs), 2), dtype=float)

    for i in range(len(modal_freqs)):

        # Arranging the tuple
        mode = int(modal_freqs[i][0])
        mode_freq = modal_freqs[i][1]

        # Save to matrix
        f_matrix[i, 0] = mode
        f_matrix[i, 1] = mode_freq

    # Save to file
    if save:
        np.savetxt(folder + name + '_frequencies_all' + '.txt', f_matrix,
                   fmt='%.3f', delimiter='\t')
        np.save(folder + name + '_frequencies_all' + '.npy', f_matrix)

    return(f_matrix)


def get_modeshapes(odb, node_set_name, save=False, name='test', folder='test'):
    """Get mode shapes from the Abaqus model.

    Note that the result is a matrix where the columns represent the mode
    shape of the frequency considered.

    The rows of each column represent a channel for a particular node, with X,
    Y and Z value respectively. The numbering of nodes is in ascending order.

    Parameters
    ----------
    odb : Abaqus object
        Abaqus object for the ODB file.
    node_set_name : str
        Name of the Abaqus node set to extract the mode shapes from.
    save : bool, optional
        When True, the natural frequencies are saved to .txt and .npy files.
    name : str, optional
        Name to be included of the saved files.
    folder : str, optional
        Name of the folder to include the saved files in.

    Returns
    -------
    phi_matrix : ndarray
        Matrix with mode shapes.
    """

    # Number of modes investigated
    number_of_frames = len(odb.steps['Modal'].frames) - 1

    # Node set
    node_set = odb.rootAssembly.nodeSets[node_set_name]
    n_sensor_nodes = len(node_set.nodes[0])
    channels = n_sensor_nodes*3

    # Mode shape matrix
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
        phi_matrix[:, i] = phi

    # Save to file
    if save:
        np.savetxt(folder + name + '_modes_all' + '.txt', phi_matrix,
                   fmt='%.10f', delimiter='\t')
        np.save(folder + name + '_modes_all' + '.npy', phi_matrix)

    return(phi_matrix)
