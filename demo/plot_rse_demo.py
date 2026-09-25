"""Plot RSE estimasi langsung vs estimasi HB pada dataset demonstrasi (30 area).

Membaca hasil langkah 15 dari demo_api.py, jadi jalankan demo_api.py lebih dulu:
    py -3.14 demo/demo_api.py
    py -3.14 demo/plot_rse_demo.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hbsaemp as hb

OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_api"
BATAS_RSE = 25  # persen

# Estimasi HB (subjek) biru, estimasi langsung (pembanding) abu-abu, batas RSE merah.
BIRU, ABU, MERAH = "#2a78d6", "#898781", "#d03b3b"
TINTA_2, GRID, SUMBU = "#52514e", "#e1e0d9", "#c3c2b7"

hasil = pd.read_csv(OUT / "langkah15_estimasi_area.csv")
df = hb.load_dataset("data_betalogitnorm")
data = hasil.merge(df[["group", "n", "deff"]], on="group").sort_values("group")

# RSE estimasi langsung dari ragam sampling proporsi dengan efek desain,
# y(1 - y) * deff / n, yaitu ragam model Beta dengan phi = n/deff - 1.
y = data["penduga_langsung"]
data["rse_langsung"] = 100 * np.sqrt(y * (1 - y) * data["deff"] / data["n"]) / y
data["rse_hb"] = data["rse_pct"]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200)
x = np.arange(len(data))
lebar = 0.4
batang_langsung = ax.bar(x - lebar / 2, data["rse_langsung"], lebar, color=ABU,
                         edgecolor="white", linewidth=0.8, label="Estimasi langsung")
batang_hb = ax.bar(x + lebar / 2, data["rse_hb"], lebar, color=BIRU,
                   edgecolor="white", linewidth=0.8, label="Estimasi HB")
garis_batas = ax.axhline(BATAS_RSE, color=MERAH, linewidth=1.5, linestyle="--",
                         label="Batas RSE 25%")

ax.set_xticks(x, data["group"])
ax.set_xlim(-0.6, len(data) - 0.4)
ax.set_ylim(0, max(data["rse_langsung"].max(), data["rse_hb"].max(), BATAS_RSE) * 1.12)
ax.set_xlabel("Area", color=TINTA_2)
ax.set_ylabel("RSE (%)", color=TINTA_2)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for sisi in ("top", "right"):
    ax.spines[sisi].set_visible(False)
for sisi in ("left", "bottom"):
    ax.spines[sisi].set_color(SUMBU)
ax.tick_params(colors=TINTA_2, labelsize=9)
ax.legend(handles=[batang_langsung, batang_hb, garis_batas], frameon=False, ncol=3,
          loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)

fig.savefig(OUT / "rse_langsung_vs_hb.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# Tabel pendamping grafik
tabel = data[["group", "penduga_langsung", "n", "deff", "rse_langsung", "rse_hb"]]
tabel.to_csv(OUT / "rse_langsung_vs_hb.csv", index=False)
print(tabel.round(2).to_string(index=False))
for nama, kolom in [("Estimasi langsung", "rse_langsung"), ("Estimasi HB", "rse_hb")]:
    print(f"{nama:18s}: rata-rata RSE {data[kolom].mean():.2f}%, "
          f"area dengan RSE <= {BATAS_RSE}%: {(data[kolom] <= BATAS_RSE).sum()} dari {len(data)}")
print(f"\nGrafik dan tabel tersimpan di {OUT}/")
