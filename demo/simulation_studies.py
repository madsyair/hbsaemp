"""Studi simulasi Monte Carlo HB Beta-logistik untuk evaluasi hbsaemp.

Membangkitkan Data Hasil Simulasi (DHS) dari model HB Beta-logistik tingkat
area (simulasi berbasis model), lalu memasang hbm_beta pada setiap replikasi
dan membandingkan hasilnya dengan nilai parameter yang ditanam.

Model pembangkit (dua lapis, sama dengan model yang diestimasi hbsaemp):
    lapis penghubung : logit(theta_i) = b0 + b1*X1 + b2*X2 + b3*X3 + v_i,
                       v_i ~ N(0, sigma_v^2)
    lapis sampling   : y_dir_i ~ Beta(theta_i*phi_i, (1 - theta_i)*phi_i),
                       phi_i = n_i/deff_i - 1

Desain:
    m = 42 area, n_i di {10, 25, 50, 100}, deff_i = 1, 1000 replikasi
    beta = (-3.20; 0.86; 0.52; -0.50), sigma_v^2 = 0.25 (Prayoga dkk., 2024)

Output di outputs/simulation/. Hasil setiap fit langsung ditulis, jadi run
dapat dihentikan (Ctrl+C) lalu dilanjutkan dengan perintah yang sama:
    hasil_area.csv          data simulasi + estimasi HB per area
    hasil_parameter.csv     estimasi, interval 95%, dan diagnostik per fit
    gagal.csv               fit yang gagal beserta pesannya
    ringkasan_area.csv      bias, RMSE, RRMSE, coverage, RSE per skenario
    ringkasan_kelas_theta.csv  kinerja per kelas proporsi sebenarnya
    ringkasan_parameter.csv bias, RMSE, coverage parameter per skenario

Jalankan dari root repo:
    py -3.14 demo/simulation_studies.py                        # run penuh
    py -3.14 demo/simulation_studies.py --reps 2 --workers 2   # uji cepat
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

import hbsaemp as hb

OUT = Path(__file__).resolve().parent.parent / "outputs" / "simulation"
SEED = 2026
# Memori worker bertambah ~30 MB per fit dan tidak dilepas; tanpa pergantian
# worker, run berjam-jam kehabisan memori. Karena itu fit dijalankan per
# kelompok (workers x 10 fit) dengan pool proses baru untuk setiap kelompok.
# max_tasks_per_child tidak dipakai: di Python 3.14.5 run macet setelah semua
# worker mencapai batasnya.
FIT_PER_WORKER = 10

# Langkah 1 (Domain area dan skenario ukuran sampel)
M = 42
SKENARIO_N = [10, 25, 50, 100]
DEFF = 1.0  # efek desain; 1 = sampling acak sederhana
N_REPLIKASI = 1000

# Langkah 2 (Parameter acuan, Prayoga dkk. 2024)
BETA = np.array([-3.20, 0.86, 0.52, -0.50])
SIGMA2_V = 0.25

PARAMETER_BENAR = {
    "beta0": BETA[0], "beta1": BETA[1], "beta2": BETA[2], "beta3": BETA[3],
    "sigma2_v": SIGMA2_V,
}
# Nama variabel posterior hbsaemp untuk setiap koefisien.
NAMA_POSTERIOR = {"beta0": "Intercept", "beta1": "X1", "beta2": "X2", "beta3": "X3"}


def bangkitkan_populasi(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Langkah 3-6: kovariat dan proporsi sebenarnya theta untuk 42 area."""
    # Langkah 3 (Kovariat area X1, X2, X3)
    x = rng.uniform(0, 1, size=(M, 3))
    # Langkah 4 (Random effect area)
    v = rng.normal(0, np.sqrt(SIGMA2_V), size=M)
    # Langkah 5 (Prediktor linear)
    eta = BETA[0] + x @ BETA[1:] + v
    # Langkah 6 (Proporsi sebenarnya melalui inverse logit)
    theta = np.exp(eta) / (1 + np.exp(eta))
    return x, theta


def bangkitkan_estimasi_langsung(
    theta: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Langkah 7-10: estimasi langsung dari distribusi Beta dan ragam samplingnya."""
    # Langkah 7 (Parameter presisi dari desain survei)
    phi = n / DEFF - 1
    # Langkah 8 (Parameter bentuk Beta)
    a = theta * phi
    b = (1 - theta) * phi
    # Langkah 9 (Estimasi langsung). Batas 1e-12 hanya pengaman numerik agar
    # tidak ada nilai tepat 0 atau 1; menyentuh ~0.1% area pada n = 10.
    y_dir = np.clip(rng.beta(a, b), 1e-12, 1 - 1e-12)
    # Langkah 10 (Ragam sampling): theta(1 - theta)/(phi + 1) = theta(1 - theta)*deff/n
    var_dir = theta * (1 - theta) / (phi + 1)
    return y_dir, var_dir


def bangkitkan_data(rep: int, n: int) -> pd.DataFrame:
    """Langkah 11: data simulasi untuk satu replikasi dan satu skenario n.

    Populasi (langkah 3-6) hanya bergantung pada replikasi, sehingga keempat
    skenario memakai theta yang sama dan hanya berbeda pada galat sampling.
    """
    x, theta = bangkitkan_populasi(np.random.default_rng([SEED, rep]))
    y_dir, var_dir = bangkitkan_estimasi_langsung(
        theta, n, np.random.default_rng([SEED, rep, n])
    )
    return pd.DataFrame({
        "area_id": np.arange(1, M + 1),
        "theta": theta,
        "y_dir": y_dir,
        "var_dir": var_dir,
        "X1": x[:, 0],
        "X2": x[:, 1],
        "X3": x[:, 2],
        "n": n,
        "deff": DEFF,
    })


def fit_dan_evaluasi(rep: int, n: int) -> tuple[pd.DataFrame, dict]:
    """Pasang hbm_beta pada satu dataset simulasi dan kumpulkan hasilnya."""
    mulai = time.perf_counter()
    df = bangkitkan_data(rep, n)

    cfg = hb.ModelConfig(draws=2000, tune=1000, chains=4, target_accept=0.95,
                         random_seed=rep * 1000 + n, progressbar=False)
    model = hb.hbm_beta(
        response="y_dir",
        auxiliary=["X1", "X2", "X3"],
        data=df,
        n="n",
        deff="deff",
        area_var="area_id",
        config=cfg,
    )
    model.fit()

    conv = hb.check_convergence(model, plot_types=[])
    tabel = conv.rhat_ess
    divergen = conv.diagnose.get("divergent")
    n_divergen = divergen["n_divergent"] if divergen else None
    max_rhat = tabel["r_hat"].max()
    min_ess_bulk = tabel["ess_bulk"].min()
    min_ess_tail = tabel["ess_tail"].min()
    param = {
        "n": n,
        "rep": rep,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "n_divergen": n_divergen,
        "konvergen": bool(
            max_rhat <= 1.01
            and min(min_ess_bulk, min_ess_tail) >= conv.ess_threshold
            and n_divergen == 0
        ),
    }

    # Rata-rata posterior dan interval kredibel 95% (kuantil 2.5% dan 97.5%).
    posterior = model.result.idata.posterior
    draws = {nama: posterior[var].values.ravel() for nama, var in NAMA_POSTERIOR.items()}
    draws["sigma2_v"] = posterior["1|area_id_sigma"].values.ravel() ** 2
    for nama, d in draws.items():
        param[nama] = float(d.mean())
        param[f"{nama}_lo"], param[f"{nama}_hi"] = np.quantile(d, [0.025, 0.975])

    est = hb.estimate_areas(model).result_table
    est = est.rename(columns={"mean": "hb_mean", "sd": "hb_sd", "rse_pct": "hb_rse_pct"})
    area = df.merge(
        est[["area_id", "hb_mean", "hb_sd", "ci_lower", "ci_upper", "hb_rse_pct"]],
        on="area_id",
    )
    area.insert(0, "rep", rep)

    param["detik"] = round(time.perf_counter() - mulai, 1)
    return area, param


def tambah_csv(df: pd.DataFrame, path: Path) -> None:
    """Tambahkan baris ke CSV; header hanya ditulis saat berkas baru dibuat.

    Berkas di folder OneDrive dapat terkunci sesaat ketika disinkronkan, jadi
    penulisan dicoba ulang sebelum galatnya dibiarkan muncul.
    """
    for _ in range(9):
        try:
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            return
        except PermissionError:
            time.sleep(3)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def mcse(per_replikasi: pd.Series) -> float:
    """Galat standar Monte Carlo dari rata-rata nilai per replikasi."""
    return per_replikasi.std(ddof=1) / np.sqrt(len(per_replikasi))


def ringkasan() -> None:
    """Hitung ringkasan per skenario dari seluruh hasil yang sudah tersimpan."""
    # drop_duplicates menangani fit yang ditulis ulang setelah run terputus.
    area = pd.read_csv(OUT / "hasil_area.csv").drop_duplicates(
        ["n", "rep", "area_id"], keep="last")
    param = pd.read_csv(OUT / "hasil_parameter.csv").drop_duplicates(
        ["n", "rep"], keep="last")
    area["rse_langsung_pct"] = 100 * np.sqrt(area["var_dir"]) / area["y_dir"]
    area["tercakup"] = (area["ci_lower"] <= area["theta"]) & (area["theta"] <= area["ci_upper"])

    # Area dalam satu replikasi saling berkaitan (berbagi estimasi beta dan
    # sigma_v), jadi MCSE dihitung dari rata-rata per replikasi.
    baris = []
    for n, d in area.groupby("n"):
        rmse_langsung = None
        for estimator, kolom, rse in [("Langsung", "y_dir", "rse_langsung_pct"),
                                      ("HB", "hb_mean", "hb_rse_pct")]:
            galat = d[kolom] - d["theta"]
            rmse = np.sqrt((galat**2).mean())
            pakai_hb = estimator == "HB"
            baris.append({
                "n": n,
                "estimator": estimator,
                "n_replikasi": d["rep"].nunique(),
                "bias": galat.mean(),
                "mcse_bias": mcse(galat.groupby(d["rep"]).mean()),
                "rmse": rmse,
                "rrmse_pct": 100 * rmse / d["theta"].mean(),
                "coverage_pct": 100 * d["tercakup"].mean() if pakai_hb else np.nan,
                "mcse_coverage_pct": (100 * mcse(d["tercakup"].groupby(d["rep"]).mean())
                                      if pakai_hb else np.nan),
                "lebar_interval": (d["ci_upper"] - d["ci_lower"]).mean() if pakai_hb else np.nan,
                # Median, karena y_dir yang sangat dekat 0 membuat RSE langsung ekstrem.
                "rse_median_pct": d[rse].median(),
                "pct_area_rse_le_25": 100 * (d[rse] <= 25).mean(),
                "rasio_rmse_hb_langsung": rmse / rmse_langsung if pakai_hb else np.nan,
            })
            rmse_langsung = rmse
    tabel_area = pd.DataFrame(baris)

    # Kinerja menurut besar proporsi sebenarnya: memperlihatkan efek shrinkage
    # pada area ekstrem yang tertutup oleh rata-rata seluruh area.
    area["kelas_theta"] = pd.cut(area["theta"], [0, 0.03, 0.06, 0.10, 1],
                                 labels=["<=0.03", "0.03-0.06", "0.06-0.10", ">0.10"])
    baris = []
    for (n, kelas), d in area.groupby(["n", "kelas_theta"], observed=True):
        baris.append({
            "n": n,
            "kelas_theta": kelas,
            "porsi_pct": 100 * len(d) / (area["n"] == n).sum(),
            "bias_langsung": (d["y_dir"] - d["theta"]).mean(),
            "bias_hb": (d["hb_mean"] - d["theta"]).mean(),
            "rmse_langsung": np.sqrt(((d["y_dir"] - d["theta"]) ** 2).mean()),
            "rmse_hb": np.sqrt(((d["hb_mean"] - d["theta"]) ** 2).mean()),
            "coverage_pct": 100 * d["tercakup"].mean(),
            "theta_di_bawah_ci_pct": 100 * (d["theta"] < d["ci_lower"]).mean(),
            "theta_di_atas_ci_pct": 100 * (d["theta"] > d["ci_upper"]).mean(),
        })
    tabel_kelas = pd.DataFrame(baris)

    baris = []
    for n, d in param.groupby("n"):
        for nama, nilai in PARAMETER_BENAR.items():
            galat = d[nama] - nilai
            tercakup = (d[f"{nama}_lo"] <= nilai) & (nilai <= d[f"{nama}_hi"])
            baris.append({
                "n": n,
                "parameter": nama,
                "nilai_benar": nilai,
                "rata_estimasi": d[nama].mean(),
                "median_estimasi": d[nama].median(),
                "bias": galat.mean(),
                "mcse_bias": mcse(galat),
                "rmse": np.sqrt((galat**2).mean()),
                "coverage_pct": 100 * tercakup.mean(),
                "mcse_coverage_pct": 100 * mcse(tercakup.astype(float)),
                "n_replikasi": len(d),
                "pct_konvergen": 100 * d["konvergen"].mean(),
            })
    tabel_param = pd.DataFrame(baris)

    tabel_area.to_csv(OUT / "ringkasan_area.csv", index=False)
    tabel_kelas.to_csv(OUT / "ringkasan_kelas_theta.csv", index=False)
    tabel_param.to_csv(OUT / "ringkasan_parameter.csv", index=False)
    print("\n=== Ringkasan estimasi area ===")
    print(tabel_area.round(4).to_string(index=False))
    print("\n=== Ringkasan menurut kelas theta ===")
    print(tabel_kelas.round(4).to_string(index=False))
    print("\n=== Ringkasan parameter ===")
    print(tabel_param.round(4).to_string(index=False))


def main() -> None:
    """Jalankan fit yang belum selesai secara paralel, lalu buat ringkasan."""
    parser = argparse.ArgumentParser(description="Studi simulasi Monte Carlo HB Beta-logistik.")
    parser.add_argument("--reps", type=int, default=N_REPLIKASI,
                        help="jumlah replikasi per skenario (default 1000)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="jumlah fit yang berjalan paralel")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Lanjutkan run sebelumnya: lewati (n, rep) yang sudah tercatat.
    selesai = set()
    if (OUT / "hasil_parameter.csv").exists():
        lama = pd.read_csv(OUT / "hasil_parameter.csv")
        selesai = set(zip(lama["n"].tolist(), lama["rep"].tolist()))
    sisa = [(n, rep) for rep in range(1, args.reps + 1) for n in SKENARIO_N
            if (n, rep) not in selesai]
    print(f"{len(selesai)} fit sudah selesai, {len(sisa)} fit tersisa "
          f"({args.workers} worker). Output: {OUT}")

    if sisa:
        mulai = time.perf_counter()
        i = 0
        kelompok = args.workers * FIT_PER_WORKER
        for awal in range(0, len(sisa), kelompok):
            # Pool baru untuk setiap kelompok, sehingga worker lama keluar dan
            # memorinya dilepas.
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                tugas = {pool.submit(fit_dan_evaluasi, rep, n): (n, rep)
                         for n, rep in sisa[awal:awal + kelompok]}
                try:
                    for fut in as_completed(tugas):
                        i += 1
                        n, rep = tugas[fut]
                        try:
                            area, param = fut.result()
                        except BrokenProcessPool:
                            # Worker mati; semua tugas berikutnya ikut gagal. Hentikan
                            # run agar dapat dilanjutkan, jangan catat sebagai fit gagal.
                            raise
                        except Exception as exc:
                            tambah_csv(pd.DataFrame([{"n": n, "rep": rep,
                                                      "error": f"{type(exc).__name__}: {exc}"}]),
                                       OUT / "gagal.csv")
                            print(f"[{i}/{len(sisa)}] n={n} rep={rep} GAGAL: {exc}")
                            continue
                        # hasil_parameter.csv ditulis terakhir karena menjadi penanda selesai.
                        tambah_csv(area, OUT / "hasil_area.csv")
                        tambah_csv(pd.DataFrame([param]), OUT / "hasil_parameter.csv")
                        menit = (time.perf_counter() - mulai) / 60
                        print(f"[{i}/{len(sisa)}] n={n} rep={rep} selesai dalam "
                              f"{param['detik']} s (konvergen={param['konvergen']}, "
                              f"total {menit:.1f} menit)")
                except BaseException:
                    # Tanpa ini, keluar dari blok with (galat atau Ctrl+C) menunggu
                    # SEMUA fit tersisa dalam kelompok dihitung tanpa disimpan.
                    pool.shutdown(cancel_futures=True)
                    raise

    if (OUT / "hasil_parameter.csv").exists():
        ringkasan()


if __name__ == "__main__":
    main()
