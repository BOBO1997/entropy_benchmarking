import time
import datetime
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from setting import *



with open("run_theoretical.pkl", "rb") as f:
    run_theoretical = pickle.load(f)
with open("run_raw_pec.pkl", "rb") as f:
    run_raw_pec = pickle.load(f)

# with open("run_pec.pkl", "rb") as f:
#     run_pec = pickle.load(f)
# with open("run_raw.pkl", "rb") as f:
#     run_raw = pickle.load(f)

energy_theoretical = run_theoretical["energy_theoretical"]

deltas_2d_raw = run_raw_pec["deltas_2d_raw"]
print(deltas_2d_raw)

deltas_2d_pec = run_raw_pec["deltas_2d_pec"]
print(deltas_2d_pec)

heatmap = np.where(deltas_2d_pec < deltas_2d_raw, 0, 1)

plt.close('all')
fig, ax = plt.subplots(
    subplot_kw={"projection": "3d"},
    dpi=200,
)
         
# Heat map
fig, ax = plt.subplots()
im = ax.imshow(deltas_2d_raw,
               origin="lower")

# Add the color bar
cbar = ax.figure.colorbar(im, ax = ax)
cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
# plt.show()
plt.savefig("heatmap_raw.png")