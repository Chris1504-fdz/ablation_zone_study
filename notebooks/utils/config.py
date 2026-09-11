"""Constants shared by the campaign notebook and the utils modules."""
import numpy as np
import torch

SEED  = 42
DTYPE = torch.double

X_COLS = ['X1', 'X2']
Y_COLS = ['Y1', 'Y2', 'Y3']
X_NAMES = {'X1': 'Carbon black wt%', 'X2': 'Fe-rich regolith wt%'}
Y_NAMES = {'Y1': 'Linear ablative rate (µm/s)', 'Y2': 'Backside temp. (°C)', 'Y3': 'Density (g/cm³)'}
Y_SHORT = {'Y1': 'Y1 Ablation (µm/s)', 'Y2': 'Y2 Backside temp (°C)', 'Y3': 'Y3 Density (g/cm³)'}

X_BOUNDS = np.array([[5.0, 1.0], [20.0, 5.0]])   # [lower; upper] per input (wt%)
NOISE_FLOOR = 1e-6      # variance floor: identical replicates give s² = 0
I369_WT = 2.0           # fixed additive wt%; resin = 100 − X1 − X2 − I-369
GRID_N = 50             # acquisition-landscape grid resolution

# colour per campaign iteration (0 = initial DOE) in every figure
IT_COLORS = {0: 'steelblue', 1: 'limegreen', 2: 'darkorange', 3: 'seagreen', 4: 'orchid', 5: 'crimson'}
