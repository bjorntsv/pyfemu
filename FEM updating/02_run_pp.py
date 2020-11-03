# -- coding utf-8 --

"""
02_run_pp.py

Postprocessing of results from the FEM updating.
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


# ---------------------------------------------------------------------------#
# DATA
# ---------------------------------------------------------------------------#

# Import data
with open('info.pkl', 'rb') as info_file:
    info = pickle.load(info_file)


# ---------------------------------------------------------------------------#
# PLOT
# ---------------------------------------------------------------------------#

# ------------------------------------ #
# SETTINGS
# ------------------------------------ #

fig_format1 = '.png'
fig_format2 = '.svg'
fig_folder = '04_Figures/'

# Matplotlib customization
plt.rcParams.update({'mathtext.fontset': 'stix'})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.linewidth': 0.75})


# ------------------------------------ #
# OBJECTIVE FUNCTION
# ------------------------------------ #

# Plot
iterations = np.arange(0, len(info['J_star_list']))

plt.figure(1, figsize=(3, 2))
plt.title('Objective function', fontweight='bold')
plt.plot(iterations, info['J_star_list'], color='black', linewidth=0.75)
plt.xticks(iterations)
plt.xlabel('Iterations')
plt.ylabel(r'$J(\Delta\theta)$')
plt.tight_layout()

# Save
plt.savefig(fig_folder + 'fig1' + fig_format1)
plt.savefig(fig_folder + 'fig1' + fig_format2)


# ------------------------------------ #
# SENSITIVITY MATRIX
# ------------------------------------ #

# Normalize
G1 = np.abs(info['G_matrices']['G1'])
G1 = G1 / np.amax(G1)
rows, cols = np.shape(G1)

# 3D plot
fig = plt.figure(2, figsize=(4, 3))
ax = plt.axes(projection='3d')
ax.set_title('Sensitivity matrix', fontweight='bold')

for j in range(rows):

    # Centering of the bars
    dx = 0.25
    dy = 0.25

    # Coordinates of the bars
    x = [0 - dx, 1 - dx, 2 - dx]
    y = [j - dy, j - dy, j - dy]
    z = np.zeros(3)

    # Size of the bars
    dx_vec = [0.5, 0.5, 0.5]
    dy_vec = [0.5, 0.5, 0.5]
    dz_vec = G1[j, :]

    # Colormap
    cmap_alt1 = plt.cm.viridis(dz_vec)
    cmap_alt2 = plt.cm.rainbow(dz_vec)

    # Plot
    bar1 = ax.bar3d(x, y, z, dx_vec, dy_vec, dz_vec, color=cmap_alt1,
                    zsort='average', alpha=1.0)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels([r'$E$', r'$M_1$', r'$M_2$'])
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels([r'$f_1$', r'$f_2$', r'$f_3$', r'$f_4$'])
ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])

# Save
plt.savefig(fig_folder + 'fig2' + fig_format1)
plt.savefig(fig_folder + 'fig2' + fig_format2)


# ------------------------------------ #
# MAC
# ------------------------------------ #

MAC_initial = info['MAC_initial']
MAC_updated = info['MAC_updated']['MAC_updated5']
MAC_updated = np.round(MAC_updated, 2)
rows, cols = np.shape(MAC_initial)

# Plot parameters
x_tick_list = np.arange(0, cols, 1)
y_tick_list = np.arange(0, rows, 1)
naming = []

for k in range(cols):
    name = str(k+1)
    naming.append(name)

# Updated MAC plot
plt.figure(3, figsize=(4, 3))
plt.imshow(MAC_updated, cmap='viridis', interpolation=None, vmin=0, vmax=1,
           origin='lower')
plt.title('MAC', fontweight='bold')
plt.colorbar()
plt.xlabel('Identified modes', fontweight='bold')
plt.ylabel('Numerical modes', fontweight='bold')
plt.xticks(x_tick_list, naming)
plt.yticks(y_tick_list, naming,
           rotation=90, va='center')

for (ii, jj), number in np.ndenumerate(MAC_updated):
    if ii == jj:
        plt.text(ii, jj, '{:0.2f}'.format(number), ha='center', va='center',
                 color='black', fontsize=6)
    else:
        plt.text(ii, jj, '{:0.2}'.format(number), ha='center', va='center',
                 color='white', fontsize=6)

plt.tight_layout()

# Save
plt.savefig(fig_folder + 'fig3' + fig_format1)
plt.savefig(fig_folder + 'fig3' + fig_format2)

plt.show()
