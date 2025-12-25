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

    th  = delta_threshold
    r   = deltas_2d_raw
    p   = deltas_2d_pec

    # 初期値：both fail
    heatmap_gap = np.full_like(r, fill_value=-1.0, dtype=float)

    # at least raw succeeds & raw wins
    mask_raw_wins = (
        (r < th) &
        (
            (p >= th) |      # pec fails
            (r <= p)         # both succeed & raw no worse (equal included)
        )
    )

    # at least pec succeeds & pec wins
    mask_pec_wins = (
        (p < th) &
        (
            (r >= th) |      # raw fails
            (p < r)          # both succeed & pec strictly better
        )
    )

    # 適用（順序はどちらでもOKだが、fail→wins の順が自然）
    heatmap_gap[mask_raw_wins] = 0.0
    heatmap_gap[mask_pec_wins] = 1.0



    plt.close('all')
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"},
        dpi=200,
    )

    # Heat map
    fig, ax = plt.subplots()
    im = ax.imshow(
        heatmap_gap,
        origin="lower",
        vmin=-1, 
        vmax=1,
        extent=[np.log10(ps_dep_global[0]), np.log10(ps_dep_global[-1]),
                np.log10(Ns_shots[0]), np.log10(Ns_shots[-1])],
        aspect="auto",
    )
    ax.set_xlabel(r'$\log_{10} p$')
    ax.set_ylabel(r'$\log_{10} N_{\mathrm{shots}}$')

    # Add the color bar
    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
    # plt.show()
    plt.savefig("heatmap_pec-wins_gap_"+str(gap)+"_Gamma_allowed_"+str(Gamma_allowed)+".png")