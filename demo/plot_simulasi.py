"""Grafik hasil studi simulasi Monte Carlo (dibaca dari outputs/simulation/).

Jalankan setelah demo/simulation_studies.py selesai:
    py -3.14 demo/plot_simulasi.py

Menghasilkan tiga gambar di outputs/simulation/:
    grafik_akurasi_area.png         RRMSE, median RSE, dan coverage HB menurut n
    grafik_parameter.png            sebaran estimasi beta dan sigma_v^2 menurut n
    grafik_coverage_kelas_theta.png coverage HB menurut kelas proporsi sebenarnya
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from simulation_studies import OUT, PARAMETER_BENAR, SKENARIO_N

# Estimasi HB biru, estimasi langsung abu-abu, batas RSE merah, garis acuan tinta gelap.
BIRU, BIRU_MUDA, ABU, MERAH = "#2a78d6", "#b7d3f6", "#898781", "#d03b3b"
TINTA, TINTA_2, GRID, SUMBU = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
# Gradasi satu warna untuk n (urutan bermakna: makin besar makin gelap).
WARNA_N = dict(zip(SKENARIO_N, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": SUMBU, "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titlesize": 10, "axes.titlecolor": TINTA,
    "axes.titlelocation": "left",
})
POSISI = np.arange(len(SKENARIO_N))
LABEL_N = [f"n = {n}" for n in SKENARIO_N]


def garis_acuan(ax: Axes, nilai: float, teks: str, warna: str, kiri: bool = False) -> None:
    """Garis acuan horizontal dengan label langsung di ujung kanan (atau kiri)."""
    ax.axhline(nilai, color=warna, linewidth=1.3, linestyle="--", zorder=1)
    ax.annotate(teks, xy=(0 if kiri else 1, nilai), xycoords=("axes fraction", "data"),
                xytext=(4 if kiri else 0, 3), textcoords="offset points",
                ha="left" if kiri else "right", va="bottom", color=TINTA_2, fontsize=8)


def simpan(fig: Figure, nama: str) -> None:
    """Simpan gambar ke outputs/simulation/ dengan latar putih."""
    fig.savefig(OUT / nama, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"tersimpan: {OUT / nama}")


# Gambar 1: akurasi dan presisi estimasi area
area = pd.read_csv(OUT / "ringkasan_area.csv")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), dpi=200)
for estimator, warna, label in [("Langsung", ABU, "Estimasi langsung"),
                                ("HB", BIRU, "Estimasi HB")]:
    d = area[area["estimator"] == estimator].set_index("n").loc[SKENARIO_N]
    gaya = dict(color=warna, linewidth=2, marker="o", markersize=6, label=label)
    axs[0].plot(POSISI, d["rrmse_pct"], **gaya)
    axs[1].plot(POSISI, d["rse_median_pct"], **gaya)
    if estimator == "HB":
        axs[2].errorbar(POSISI, d["coverage_pct"], yerr=1.96 * d["mcse_coverage_pct"],
                        capsize=3, **gaya)
axs[0].set(title="(a) RRMSE", ylabel="RRMSE (%)", ylim=(0, None))
axs[1].set(title="(b) Median RSE", ylabel="RSE (%)", ylim=(0, None))
garis_acuan(axs[1], 25, "Batas RSE 25%", MERAH, kiri=True)
axs[2].set(title="(c) Coverage interval HPD 95% (HB)", ylabel="Coverage (%)", ylim=(88, 98))
garis_acuan(axs[2], 95, "Nominal 95%", TINTA)
for ax in axs:
    ax.set_xticks(POSISI, LABEL_N)
    ax.set_xlabel("Ukuran sampel per area")
fig.legend(handles=axs[0].get_lines(), loc="lower left", bbox_to_anchor=(0.05, 0.98),
           ncol=2, frameon=False)
fig.tight_layout()
simpan(fig, "grafik_akurasi_area.png")

# Gambar 2: sebaran estimasi parameter (rata-rata posterior per replikasi)
param = pd.read_csv(OUT / "hasil_parameter.csv")
judul = {"beta0": r"$\beta_0$", "beta1": r"$\beta_1$", "beta2": r"$\beta_2$",
         "beta3": r"$\beta_3$", "sigma2_v": r"$\sigma_v^2$"}
fig, axs = plt.subplots(1, len(PARAMETER_BENAR), figsize=(14, 3.6), dpi=200)
for ax, (nama, nilai) in zip(axs, PARAMETER_BENAR.items()):
    ax.boxplot([param.loc[param["n"] == n, nama] for n in SKENARIO_N], positions=POSISI,
               widths=0.55, whis=(2.5, 97.5), showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=BIRU_MUDA, edgecolor=BIRU),
               whiskerprops=dict(color=BIRU), capprops=dict(color=BIRU),
               medianprops=dict(color=TINTA, linewidth=1.2))
    ax.axhline(nilai, color=MERAH, linewidth=1.3, linestyle="--", zorder=1)
    ax.set_title(judul[nama])
    ax.set_xticks(POSISI, [str(n) for n in SKENARIO_N])
    ax.set_xlabel("n")
fig.legend(handles=[Line2D([], [], color=MERAH, linestyle="--", label="Nilai acuan")],
           loc="lower left", bbox_to_anchor=(0.04, 0.98), frameon=False)
fig.tight_layout()
simpan(fig, "grafik_parameter.png")

# Gambar 3: coverage HB menurut kelas proporsi sebenarnya
kelas = pd.read_csv(OUT / "ringkasan_kelas_theta.csv")
urutan = ["<=0.03", "0.03-0.06", "0.06-0.10", ">0.10"]
label_kelas = ["≤ 0,03", "0,03–0,06", "0,06–0,10", "> 0,10"]
fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
for n in SKENARIO_N:
    d = kelas[kelas["n"] == n].set_index("kelas_theta").loc[urutan]
    ax.plot(np.arange(len(urutan)), d["coverage_pct"], color=WARNA_N[n], linewidth=2,
            marker="o", markersize=6, label=f"n = {n}")
garis_acuan(ax, 95, "Nominal 95%", TINTA)
ax.set_xticks(np.arange(len(urutan)), label_kelas)
ax.set(xlabel=r"Kelas proporsi sebenarnya $\theta$", ylabel="Coverage HB (%)", ylim=(75, 100))
ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
simpan(fig, "grafik_coverage_kelas_theta.png")
