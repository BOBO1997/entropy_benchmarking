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

deltas_3d_pec = run_raw_pec["deltas_3d_pec"]
print(deltas_3d_pec)

# ================================
# Parameters for classification
# ================================
delta_threshold = 0.05           # success / fail threshold
eps_relative = 1e-6              # relative tolerance for "almost equal"


for ith_Gamma_allowed, Gamma_allowed in enumerate(Gammas_allowed):

    deltas_2d_pec = deltas_3d_pec[:, :, ith_Gamma_allowed]

    # heatmap encoding:
    #  -1.0 : both fail
    #   0.0 : raw wins
    #   0.5 : almost equal (both succeed)
    #   1.0 : PEC wins
    heatmap_gap = np.zeros_like(deltas_2d_raw, dtype=float)

    # ----------------------------
    # Masks
    # ----------------------------

    # (1) both fail
    mask_both_fail = (
        (deltas_2d_raw >= delta_threshold) &
        (deltas_2d_pec >= delta_threshold)
    )

    # (2) both succeed
    mask_both_succeed = (
        (deltas_2d_raw < delta_threshold) &
        (deltas_2d_pec < delta_threshold)
    )

    # (3) almost equal (relative comparison, only in success region)
    mask_almost_equal = (
        mask_both_succeed &
        (np.abs(deltas_2d_raw - deltas_2d_pec)
         <= eps_relative * np.maximum(deltas_2d_raw, deltas_2d_pec))
    )

    # (4) PEC strictly wins (success region only)
    mask_pec_wins = (
        mask_both_succeed &
        (~mask_almost_equal) &
        (deltas_2d_pec < deltas_2d_raw)
    )

    # (5) raw strictly wins (success region only)
    mask_raw_wins = (
        mask_both_succeed &
        (~mask_almost_equal) &
        (deltas_2d_raw < deltas_2d_pec)
    )

    # ----------------------------
    # Apply labels (priority order)
    # ----------------------------
    heatmap_gap[mask_both_fail]   = -1.0
    heatmap_gap[mask_raw_wins]    =  0.0
    heatmap_gap[mask_almost_equal]=  0.5
    heatmap_gap[mask_pec_wins]    =  1.0


    plt.close('all')
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"},
        dpi=200,
    )

    # Heat map
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_gap,
                origin="lower")

    # Add the color bar
    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
    # plt.show()
    plt.savefig("heatmap_pec-wins_gap_"+str(gap)+"_Gamma_allowed_"+str(Gamma_allowed)+".png")