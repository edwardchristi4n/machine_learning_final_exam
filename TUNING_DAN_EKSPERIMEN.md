# Dokumen Tuning & Eksperimen — KelpAdam UAS Machine Learning

Dokumen ini merangkum **semua angka dan variabel yang dipakai untuk tuning hyperparameter** di Notebook A (Klasifikasi) dan Notebook B (Regresi), **alasan setiap angka dipilih**, **alasan memakai GridSearchCV** sebagai metode tuning, serta **alasan setiap eksperimen wajib dijalankan** untuk menyusun laporan UAS.

Semua angka di dokumen ini dapat ditelusuri ulang ke file output di `figures_and_outputs/` (CSV tersimpan otomatis setiap kali notebook dijalankan).

---

## 1. Konstanta eksperimen yang mengikat (sama di kedua notebook)

| Variabel | Nilai | Alasan |
|---|---|---|
| `SEED` (`random_state`) | **42** | Nilai wajib dari spesifikasi UAS. Menjamin hasil identik setiap kali dijalankan ulang (reproducible). |
| `TEST_SIZE` | **0.2** | Wajib dari spesifikasi: train 80% : test 20%. |
| `cv` (A) | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | 5 lipatan; **stratified** karena targetnya 5 kategori dan test hanya ±39 baris — tanpa stratifikasi lipatan bisa kehilangan kelas kecil (Seri). |
| `cv` (B) | `KFold(n_splits=5, shuffle=True, random_state=42)` | 5 lipatan; tanpa stratifikasi karena target IPK kontinu (stratifikasi untuk kategori). Kesesuaian split dibuktikan lewat uji KS (p = 0,81). |
| `n_estimators` RF | **500** (tidak di-tuning) | Menambah jumlah pohon hanya menurunkan **variansi**, bukan memperbaiki bias. Diverifikasi terpisah di B5.2 dengan 100/300/500/1000 pohon: ≥300 hasil identik, 1000 dua kali lebih lambat. 500 = titik stabil dengan biaya wajar. |

Kenapa **5-fold**? Data kecil (train 153 / 138 baris). Lipatan lebih sedikit (mis. 3) membuat data latih per model lebih banyak tetapi evaluasi lebih kasar; lipatan lebih banyak (mis. 10) membuat tiap lipatan validasi terlalu kecil (±15 baris) sehingga skor goyang. 5 adalah keseimbangan standar: tiap model dilatih di 80% train dan tiap lipatan validasi berisi ±30 baris. Objek `cv` yang **sama** dipakai untuk semua model dan GridSearch, sehingga semua kandidat diuji pada potongan data yang persis sama (perbandingan adil).

---

## 2. Kenapa metode tuning = GridSearchCV (bukan RandomizedSearch / manual)

| # | Alasan | Penjelasan |
|---|---|---|
| 1 | **Ruang pencarian kecil, exhaustive terjangkau** | Grid terbesar hanya 40 kombinasi (SVM) → 40 × 5 lipatan = 200 training, selesai dalam hitungan detik–menit di data 153 baris. RandomizedSearch dibuat untuk grid ratusan–ribuan kombinasi; di sini justru membuang jaminan menemukan kombinasi terbaik. |
| 2 | **Deterministik & adil** | Grid menguji **semua** kombinasi pada **lipatan CV yang identik** (`cv` + `random_state` sama), jadi perbandingan antar kombinasi benar-benar apples-to-apples dan dapat direproduksi siapa pun. |
| 3 | **Terintegrasi dengan Pipeline (bebas leakage)** | Parameter ditulis dengan prefix `model__`, sehingga `StandardScaler` di-fit ulang di setiap lipatan untuk **setiap** kombinasi. Tuning manual (fit scaler sekali lalu loop parameter) berisiko bocor data validasi ke scaler. |
| 4 | **Jawaban eksplisit untuk laporan** | Karena grid exhaustive, kita bisa menyatakan: *"seluruh kemungkinan dalam grid telah diuji"* — dibuktikan tabel lengkap 40/24/30 baris di `figures_and_outputs/A5_2…`, `A5_3…`, `B5_2…`. |

**Metrik pemilihan (refit):** Notebook A → `refit="macro_F1"` (kelas kecil Seri & Read/Write ikut berbobot); Notebook B → `refit="MAE"` (satuan poin IPK, mudah diinterpretasi).

---

## 3. Notebook A (Klasifikasi) — angka tuning dan alasannya

### 3.1 SVM — `GRID_SVM`, 40 kombinasi × 5 lipatan = 200 training

```python
GRID_SVM = [
    {"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1, 10],
     "model__class_weight": [None, "balanced"]},                          # 8 kombinasi
    {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100],
     "model__gamma": ["scale", 0.001, 0.01, 0.1],
     "model__class_weight": [None, "balanced"]},                          # 32 kombinasi
]
```

| Parameter | Nilai dicoba | Kenapa segitu |
|---|---|---|
| `kernel` | `linear`, `rbf` | Label = modalitas dengan skor tertinggi dari 4 penjumlahan + kelas Seri saat skor seri → batas antar kelas tidak sepenuhnya lurus. `linear` menguji "cukulkah garis lurus", `rbf` menguji batas melengkung. Dua kernel diuji sebagai grid **terpisah** karena `gamma` hanya relevan untuk rbf. |
| `C` (linear) | 0.01, 0.1, 1, **10** | C = kekuatan hukum kesalahan latih. Efeknya lintas **orde besaran** → wajib grid skala logaritma. C kecil = terlalu direm, C besar = menghafal. **1 = default sklearn**, selalu masuk grid sebagai patokan apakah tuning menolong. |
| `C` (rbf) | 0.1, 1, **10**, 100 | Sama, tetapi interaksinya dengan gamma perlu rentang lebih lebar. Hasil: terbaik **C = 1** → tepat di **tengah** range, artinya grid sudah "mengurung" optimum (tidak perlu diperlebar). C = 0.1 hancur (≈0.07, seperti baseline) dan C = 10–100 menurun (0.59–0.62). |
| `gamma` | `scale`, 0.001, 0.01, 0.1 | `scale` = default sklearn; setelah `StandardScaler`, gamma efektifnya = 1/64 ≈ **0.016** — jatuh di antara 0.01 dan 0.1 dalam grid. Tiga dekade log (0.001–0.1) mem-bracket default itu. gamma = 0.1 buruk (≈0.15): tiap titik hanya "melihat" dirinya sendiri → menghafal. |
| `class_weight` | `None`, `"balanced"` | EDA A3.1 menemukan imbalance **ringan** (rasio 1.7, Seri 15.6%). `balanced` memberi bobot lebih ke kelas kecil. Keputusan **diserahkan ke CV** karena macro-F1 (metrik utama) sensitif terhadap kelas kecil. Hasil: balanced menang tipis 0.667 vs 0.658 — konsisten dengan imbalance yang memang ringan. |

**Hasil lengkap seluruh kemungkinan** (macro-F1 CV, 5 lipatan):

| Kandidat | macro-F1 CV | Keterangan |
|---|---|---|
| Baseline (kelas terbanyak) | 0.085 | Patokan bawah |
| SVM default | 0.658 | `C=1, gamma=scale, rbf` — default sudah dekat optimal |
| **SVM tuning (terbaik)** | **0.667** (SD 0.070) | `rbf, C=1, gamma=scale, balanced` |
| RF default | 0.559 | |
| **RF tuning (terbaik)** | **0.590** (SD 0.042) | `max_depth=None, min_samples_leaf=2, max_features=sqrt, balanced` |

**Catatan kejujuran untuk laporan:** selisih tuning vs default (0.009) jauh lebih kecil daripada SD antar lipatan (±0.07) → tuning di SVM terutama **memastikan tidak ada setelan lain yang jauh lebih baik**, bukan melompatkan performa. Skor terbaik dari 40 percobaan juga sedikit optimis (efek "juara dipilih dari banyak peserta") — angka final yang diakui adalah hasil **data test di A6** yang belum pernah disentuh.

### 3.2 Random Forest — `GRID_RF`, 24 kombinasi × 5 lipatan = 120 training

```python
GRID_RF = {
    "model__max_depth": [None, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.3],
    "model__class_weight": [None, "balanced"],
}   # 2 × 3 × 2 × 2 = 24 kombinasi
```

| Parameter | Nilai dicoba | Kenapa segitu |
|---|---|---|
| `max_depth` | `None`, `10` | `None` = default (pohon bebas tumbuh); `10` = versi direm. Menguji apakah membatasi kedalaman membantu pada data 153 baris. Hasil: skor identik → GridSearch ambil `None`. |
| `min_samples_leaf` | 1, 2, **4** | Rem anti-menghafal: minimal berapa responden per daun. 1 = default (daun bisa berisi 1 orang = hafal). Nilai kecil bertingkat (bukan loncatan besar) karena data kecil. Terbaik **2** → di tengah range. |
| `max_features` | `sqrt`, `0.3` | `sqrt` = default klasifikasi (√64 = 8 fitur per percabangan); `0.3` = 19 fitur. Menguji apakah pohon yang melihat lebih banyak fitur lebih baik (ternyata tidak — semua 10 besar memakai `sqrt`; pohon beragam lebih tahan noise). |
| `class_weight` | `None`, `balanced` | Alasan sama dengan SVM; 4 kombinasi teratas semuanya `balanced`. |
| `n_estimators` | 500 (tetap) | Lihat §1. |

Kenapa RF kalah dari SVM (0.590 vs 0.667)? Label = *jumlah* 16 indikator per modalitas lalu dibandingkan. Pohon hanya bisa memotong **satu fitur per langkah** ("Q01_K = 1?"), sehingga butuh sangat banyak percabangan untuk meniru "jumlah K > jumlah V" dengan 153 contoh. Ini dijelaskan di notebook (A5.3) dan wajib masuk laporan.

---

## 4. Notebook B (Regresi) — angka tuning dan alasannya

Data: 173 baris (19 tanpa IPK dikeluarkan, tidak diimputasi) → 138 train : 35 test.

### 4.1 Random Forest — `GRID_RF`, 30 kombinasi × 5 lipatan = 150 training

```python
GRID_RF = {
    "model__min_samples_leaf": [1, 3, 5, 10, 20],
    "model__max_features": ["sqrt", 0.3, 1.0],
    "model__max_depth": [None, 5],
}   # 5 × 3 × 2 = 30 kombinasi
```

| Parameter | Nilai dicoba | Kenapa segitu |
|---|---|---|
| `min_samples_leaf` | 1, **3**, 5, 10, 20 | **Rem paling penting.** EDA B3.5 sudah memperingatkan: 64 fitur vs 138 data train + sinyal lemah → RF default menghafal (MAE train 0.125 vs CV 0.354). Rentang 1–20 dibuat **bertahap** supaya kurva "rem" terlihat (grafik B5.2): error train naik, error CV turun, lalu landai di ≈0.336 = baseline. Terbaik **3** → di tengah range, bukan di tepi. |
| `max_features` | `sqrt`, `0.3`, `1.0` | `1.0` = **default regresi** sklearn (semua 64 fitur); `sqrt` = 8; `0.3` = 19. Default wajib masuk grid sebagai pembanding. Hasil: `sqrt` sedikit terbaik — pohon beragam sedikit lebih tahan noise. |
| `max_depth` | `None`, **5** | `None` = default; `5` = rem kuat (IPK hampir tidak punya sinyal, pohon dalam hanya menangkap noise). Hasil: `5` menang 0.0001 poin — praktis seri, dan diinterpretasi jujur sebagai seri. |
| `n_estimators` | 500 (tetap) + cek terpisah 100/300/500/1000 | Diverifikasi di sel terpisah dengan CV di train (bukan test, sesuai aturan Panduan UAS §4.3): 300/500/1000 identik (0.336), 100 sedikit lebih buruk (0.339), 1000 dua kali lebih lambat. |

**Hasil seluruh kemungkinan (MAE CV, makin kecil makin baik):**

| Kandidat | MAE CV | R² CV |
|---|---|---|
| Baseline (rata-rata) | 0.336 | −0.081 |
| Linear Regression | 0.507 | −1.460 (overfit berat) |
| RF default | 0.354 | −0.153 (menghafal) |
| **RF tuning (terbaik)** | **0.336** | −0.073 |
| Ridge (opsional) | 0.336 | −0.079 |

10 kombinasi teratas hanya berbeda maksimal 0.002 poin IPK → semua setelan "rem" praktis sama bagusnya; GridSearch hanya mengambil angka terkecil.

### 4.2 Ridge (opsional, pembanding) — 21 kombinasi × 5 lipatan = 105 training

```python
{"model__alpha": np.logspace(-1, 4, 21)}   # 10^-1 s.d. 10^4, 21 titik berjarak logaritmik
```

| Parameter | Nilai dicoba | Kenapa segitu |
|---|---|---|
| `alpha` | 10⁻¹ … 10⁴ (21 titik, skala log) | Kekuatan rem koefisien; efeknya lintas orde → wajib log grid. 21 titik = resolusi ¼ dekade, cukup halus tanpa boros. Hasil: alpha ≈ **3162** (10^3.5) — rem nyaris maksimal, koefisien tertekan mendekati 0, Ridge praktis menebak rata-rata. |

Eksperimen ini menjawab pertanyaan diagnosis: **apakah Linear Regression buruk karena overfitting atau salah kode?** Jawabannya overfitting (65 parameter untuk 110 baris latih per lipatan): begitu diberi rem, error turun dari 0.507 → 0.336. Ridge sengaja **tidak ikut dipilih** sebagai model akhir (model wajib hanya LR dan RF).

---

## 5. Prinsip umum pemilihan angka (berlaku untuk semua grid di atas)

1. **Default sklearn selalu masuk grid** (C=1, gamma=scale, leaf=1, max_features=sqrt/1.0, max_depth=None, class_weight=None). Tujuannya: laporan bisa menjawab "apakah tuning benar-benar menolong?" dengan bukti, bukan asumsi.
2. **Parameter berorde-besar (C, gamma, alpha) → grid skala logaritma.** Melipat 10 nilai C tidak berguna; melintasi 4–5 dekade yang penting.
3. **Grid dirancang agar optimum jatuh di tengah, bukan di tepi.** Kalau nilai terbaik ada di tepi range, artinya grid belum cukup lebar dan harus diperluas. Kenyataannya: C=1 (tengah), gamma=scale≈0.016 (di antara 0.01 dan 0.1), leaf=2/3 (tengah) — semua ter-bracket.
4. **Grid dibuat kecil karena data kecil.** SD antar lipatan (±0.04–0.07) jauh lebih besar dari selisih antar kombinasi teratas. Grid raksasa hanya menghasilkan "pemenang kebetulan" (winner's curse), bukan model lebih baik.
5. **Pembanding "rem" vs "tanpa rem"** dipasangkan berpasangan (None/10, 1/2/4, 1/3/5/10/20) supaya efek satu parameter terbaca dari hasilnya.
6. **Semua seleksi hanya memakai data train + CV.** Data test tidak pernah disentuh saat tuning (aturan Panduan UAS; juga alasan `n_estimators` dicek dengan CV, bukan hasil test seperti di Modul 5 praktikum).

---

## 6. Kenapa eksperimen-eksperimen ini WAJIB dijalankan untuk laporan

Alasan utama: **setiap klaim di laporan harus punya angka dan file bukti yang bisa ditelusuri ulang**, dan notebook dirancang agar `Restart & Run All` menghasilkan angka yang sama. Semua tabel/grafik tersimpan otomatis ke `figures_and_outputs/` — inilah sumber kutipan angka laporan.

### Notebook A (Klasifikasi)

| Eksperimen | Bukti yang dihasilkan | Klaim laporan yang bergantung padanya |
|---|---|---|
| A1.4 audit 216 kolom + `assert` | X = tepat 64 indikator VARK; ID/IPK/IPS/skor VARK tidak bocor | "Model bebas data leakage" — tanpa sel ini klaimnya kosong |
| A2.0 re-encoding VARK dari teks + banding identik | 0 selisih vs 64 kolom UTS | Pengodean benar sesuai panduan & fungsi yang sama akan dipakai Streamlit |
| A2.1–A2.3 validasi & keputusan cleaning | 0 kosong, 0 duplikat ID, 2 pola-X-identik & 15 straightliner dipertahankan + alasan | Keputusan preprocessing bisa dipertanggungjawabkan, bukan asal buang |
| A3.1 distribusi label | rasio imbalance 1.7; baseline accuracy 26.6% | Dasar pemilihan stratified split, macro-F1, dan `class_weight` di tuning |
| A3.2 selisih skor tertinggi–kedua | 39% responden di selisih 0–1 | Prediksi "kesalahan menumpuk di Seri" yang dicek di A6 — rantai argumen EDA→model→error |
| A4.1 split + tabel sebaran | 153:39, label terwakili (6–10 per kelas di test) | Split sesuai spesifikasi & adil |
| A4.2 pipeline + lipatan CV | Scaler di dalam pipeline; 5 lipatan stratified | Bukati teknis "bebas leakage" saat training |
| A5.1 baseline vs default | 0.085 vs 0.658 vs 0.559 | "Kedua model wajib mengalahkan baseline dengan margin jelas" |
| **A5.2 GridSearch SVM (200 fit)** | `A5_2_gridsearch_svm.csv` (40 baris) + heatmap | "Seluruh kemungkinan diuji"; parameter terbaik + bukti tidak ada setelan lebih baik |
| **A5.3 GridSearch RF (120 fit)** | `A5_3_gridsearch_rf.csv` (24 baris) | Idem untuk RF + penjelasan kenapa RF kalah |
| A5.4 perbandingan + F1 per kelas | `A5_4_perbandingan_cv.csv`, `A5_4_f1_per_kelas_cv.csv` | Basis pemilihan model (SVM 0.667 vs RF 0.590) + kelas sulit (Seri 0.40) |
| A5.5 pemilihan model | Aturan ditetapkan sebelum lihat hasil | Model dipilih tanpa "mengintip" — kriteria penilaian metodologi |
| A6 evaluasi test *(belum dikerjakan)* | confusion matrix, error per selisih skor | Verifikasi terakhir: CV → test konsisten? Wajib sebelum angka masuk laporan |
| A7 simpan model *(belum dikerjakan)* | `Model_klasifikasi.joblib` + uji muat ulang | Model yang dievaluasi = model yang dikumpulkan/dipakai Streamlit |

### Notebook B (Regresi)

| Eksperimen | Bukti yang dihasilkan | Klaim laporan yang bergantung padanya |
|---|---|---|
| B1.4 audit 216 kolom + `assert` | X hanya 64 indikator; 9 kolom IPS & IPK dilarang masuk | "Bebas leakage" — IPS membocorkan IPK, jadi ini kritis |
| B2.3 asal 19 IPK kosong | Semua tanpa riwayat IPS; **tidak diimputasi** | Kepatuhan aturan panduan (IPK hilang ≠ boleh diisi rerata) |
| B2.4 IPK ekstrem | 8 ekstrem bawah dipertahankan + konsisten dengan IPS | Keputusan defensible + prediksi "error terbesar di IPK rendah" (dibuktikan B6.4) |
| B3.4 korelasi fitur–IPK | Korelasi terbesar 0.20, tidak signifikan setelah koreksi | Dasar ekspektasi "model sulit mengalahkan baseline" — bukan alasan tuning gagal |
| B4.1–B4.2 split + uji KS | 138:35; p = 0.81; catatan 3 IPK terendah ada di train | Split representatif + batas interpretasi hasil test |
| B5.1 baseline & LR | MAE 0.336 vs 0.507; 17/138 tebakan > 4.0 | LR overfit (65 parameter vs 110 baris) — temuan wajib dilaporkan |
| **B5.2 GridSearch RF (150 fit)** | `B5_2_gridsearch_rf.csv` (30 baris) + grafik "rem" | "Seluruh kemungkinan diuji"; bukti rem menghentikan hafalan (grafik train vs CV) |
| B5.2 cek `n_estimators` | 100/300/500/1000 identik ≥300 | Justifikasi angka 500 yang tidak di-tuning |
| B5.3 Ridge | alpha 3162 → MAE 0.336 | Diagnosis: masalah LR = overfitting, bukan salah kode |
| **B5.4 uji permutasi** | p = 0.29 | Bukti statistik terkuat: performa RF tidak berbeda dari kebetulan |
| B5.5 perbandingan semua model | `B5_5_perbandingan_cv.csv` | Basis pemilihan RF sebagai model akhir |
| B6.1 test + bootstrap CI | MAE 0.345 vs baseline 0.356; CI −0.029…+0.007 mencakup 0 | Kesimpulan akhir: RF ≈ baseline, perbedaan tak signifikan |
| B6.2–B6.6 grafik, residual, error per kelompok, 5 error terbesar, permutation importance | File `B6_*.csv/png` | Analisis kesalahan sesuai panduan (IPK < 3.0 MAE ±0.65; tebakan menumpuk 3.22–3.48) |
| B7 simpan + muat ulang + model card | `Model_regresi.joblib`, `model_card_regresi.json` | Reproduksibilitas & model yang dievaluasi = model yang dikumpulkan |

### Ringkasan 3 alasan "wajib dijalankan"

1. **Ketertelusuran angka (traceability).** Setiap angka di laporan (0.667, 0.336, p = 0.29, dll.) harus punya file CSV/PNG sumbernya. Tanpa menjalankan notebook, angka laporan tidak bisa diverifikasi ulang.
2. **Rantai bukti klaim metodologi.** Klaim "bebas leakage" (audit kolom + pipeline), "tuning memakai seluruh kemungkinan" (tabel grid lengkap), "model dipilih tanpa mengintip test" (urutan kerja B5 → B6), dan "RF ≈ baseline" (permutasi + bootstrap) hanya berdiri kalau sel-selnya benar-benar dijalankan berurutan.
3. **Kepatuhan rubrik & panduan.** Spesifikasi UAS mengikat `random_state=42`, split 80:20, dua model wajib, baseline, tuning di train saja, evaluasi test sekali di akhir, dan model tersimpan — semuanya diverifikasi lewat `assert` dan file output saat notebook dijalankan penuh (`Restart & Run All`).

---

*Dokumen ini dibuat sebagai pendamping laporan; tidak mengubah kode notebook apa pun. Semua angka bersumber dari output notebook per 25 November 2026.*
