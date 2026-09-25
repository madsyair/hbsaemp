"""Demonstrasi pemanggilan API hbsaemp secara berurutan (alur Gambar 11).

Dataset : data_betalogitnorm (30 area, mengikuti dataset hbsaems di R)
Model   : Beta logit-normal dengan presisi dari desain survei (n, deff)

Langkah yang didemonstrasikan:
    2.  Memuat dataset                         -> load_dataset()
    3.  Spesifikasi model (tingkat pemula)     -> hbm_beta()
    4.  Pengecekan prior                       -> check_prior()
    6.  Pemasangan model                       -> model.fit()
    7.  Diagnostik konvergensi                 -> check_convergence()
    10. Perbandingan dengan model create_model -> create_model() + compare_models()
    14. Estimasi area                          -> estimate_areas()
    15. Hasil estimasi area                    -> result_table

Jalankan dari root repo:
    py -3.14 demo/demo_api.py
"""
from datetime import datetime
from pathlib import Path

import hbsaemp as hb

waktu_mulai = datetime.now()

# Output selalu ke <root repo>/outputs/demo_api, dari folder mana pun skrip dijalankan.
OUT = Path(__file__).resolve().parent.parent / "outputs" / "demo_api"
OUT.mkdir(parents=True, exist_ok=True)

# cfg_beta untuk config model beta (tingkat pemula)
# cfg_cm untuk config model create_model (tingkat menengah)
cfg_beta = hb.ModelConfig(draws=2000, tune=1000, chains=4, target_accept=0.95, random_seed=42)
cfg_cm = hb.ModelConfig(draws=3000, tune=1500, chains=6, target_accept=0.95, random_seed=42)


# Langkah 2 (Memuat dataset)
print("\n=== Langkah 2: Memuat dataset ===")
df = hb.load_dataset("data_betalogitnorm")
print(f"Ukuran data: {df.shape[0]} area x {df.shape[1]} kolom")
print(df.head())


# Langkah 3 (Spesifikasi model melalui hbm_beta)
print("\n=== Langkah 3: Spesifikasi model (hbm_beta) ===")
model_beta = hb.hbm_beta(
    response="y",
    auxiliary=["x1", "x2", "x3"],
    data=df,
    n="n",
    deff="deff",
    area_var="group",
    config=cfg_beta,
)
print("Formula yang dirakit:", model_beta.formula)
print(model_beta.summary())


# Langkah 4 (Pengecekan prior)
print("\n=== Langkah 4: Pengecekan prior ===")
prior = hb.check_prior(model_beta)
print(prior.summary())
print(prior.prior_summary)
if prior.prior_predictive_plot is not None:
    prior.prior_predictive_plot.savefig(OUT / "langkah04_prior_predictive.png")
print("Model sudah dipasang?", model_beta.is_fitted)


# Langkah 6 (Pemasangan model)
print("\n=== Langkah 6: Pemasangan model (fit) ===")
model_beta.fit()
print(model_beta.summary())


# Langkah 7 (Diagnostik konvergensi)
print("\n=== Langkah 7: Diagnostik konvergensi ===")
conv = hb.check_convergence(model_beta, plot_types=["trace", "rhat"])
print(conv.summary())
print(conv.rhat_ess[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]])
for nama, fig in conv.plots.items():
    fig.savefig(OUT / f"langkah07_{nama}.png")


# Langkah 10 (Perbandingan dengan model yang dibangun melalui create_model)
print("\n=== Langkah 10: Perbandingan model ===")
model_cm = hb.create_model(
    "y ~ x1 + x2 + (1|group)",
    family="beta",
    data=df,
    n="n",
    deff="deff",
    config=cfg_cm,
)
print(model_cm.summary())
model_cm.fit()
print(hb.check_convergence(model_cm, plot_types=[]).summary())

cmp = hb.compare_models([model_beta, model_cm])
print(cmp.summary())
print(cmp.comparison_table)
if cmp.compare_plot is not None:
    cmp.compare_plot.savefig(OUT / "langkah10_compare.png")

kandidat = {"model_0": model_beta, "model_1": model_cm}
nama_terbaik = cmp.comparison_table.index[0]
model_terbaik = kandidat[nama_terbaik]
print(f"Model terbaik menurut ELPD-LOO: {nama_terbaik} ({model_terbaik.formula})")


# Langkah 14 (Estimasi area)
print("\n=== Langkah 14: Estimasi area ===")
est = hb.estimate_areas(model_terbaik)
print(est.summary())


# Langkah 15 (Hasil estimasi area)
print("\n=== Langkah 15: Hasil estimasi area ===")
hasil = est.result_table.merge(df[["group", "y", "theta"]], on="group")
hasil = hasil.rename(columns={"y": "penduga_langsung", "theta": "nilai_sebenarnya"})
print(hasil[["group", "penduga_langsung", "mean", "ci_lower", "ci_upper",
             "rse_pct", "nilai_sebenarnya"]].round(4).to_string(index=False))
hasil.to_csv(OUT / "langkah15_estimasi_area.csv", index=False)
print(f"\nHasil dan gambar tersimpan di {OUT}/")


# Waktu eksekusi demo
waktu_selesai = datetime.now()
durasi = (waktu_selesai - waktu_mulai).total_seconds()
print(f"\nWaktu mulai   : {waktu_mulai:%Y-%m-%d %H:%M:%S}")
print(f"Waktu selesai : {waktu_selesai:%Y-%m-%d %H:%M:%S}")
print(f"Durasi        : {durasi:.1f} detik ({durasi / 60:.1f} menit)")
