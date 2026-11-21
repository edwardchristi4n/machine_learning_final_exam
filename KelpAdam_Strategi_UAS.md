# Strategi UAS Pembelajaran Mesin – KelpAdam

**Proyek A:** klasifikasi preferensi belajar · **Proyek B:** regresi prediksi IPK
Disusun 20 November 2026 (struktur folder diperbarui 21 November 2026) dari: Spesifikasi Soal UAS (dengan rubrik), Panduan UAS Klasifikasi, Panduan UAS Regresi, notebook UTS `KelpAdam_UTS_DataPreparation_EDA.ipynb`, dan dataset `KelpAdam_survey_prepared.csv` (192 baris × 216 kolom).

> **Catatan penting tentang angka di dokumen ini.** Semua angka eksperimen di sini berasal dari **cross-validation (CV) pada bagian training saja** (split 80:20, `random_state=42`). Test set **sengaja belum dilihat**, supaya pemilihan framework, model, dan parameter tidak bocor dari data test. Angka ini adalah eksplorasi awal untuk menyusun strategi. Angka resmi untuk laporan harus berasal dari notebook kelompok sendiri (spesifikasi: *"Tidak diperkenankan menyajikan output/angka eksperimen tanpa bukti notebook"*).

---

## 0. Ringkasan keputusan

| Keputusan | Pilihan yang disarankan | Alasan singkat |
|---|---|---|
| Framework Proyek A | **VARK** (64 indikator 0/1 → `VARK_label`) | Tidak terdampak straightliner, kelas cukup seimbang (rasio 1,7), 64 indikator sudah disiapkan di UTS, analisis kesalahan kelas *Seri* menarik. Alternatif kuat: Felder–Silverman (lihat §3) |
| Framework Proyek B | **VARK** (64 indikator → `IPK`) | Semua framework sama-sama tidak punya sinyal untuk IPK (§3.2), jadi pilih yang paling bersih dan satu skema input dengan Proyek A |
| Baris dipakai | A: 192 baris · B: 173 baris (IPK valid) | Tidak ada baris yang dibuang tambahan; 19 IPK kosong tidak diimputasi |
| Straightliner (15) | **Dipertahankan** di kedua proyek | Flag-nya dari 88 item Likert; jawaban VARK mereka tetap bervariasi (16–64 pilihan), jadi tidak ada dasar membuangnya. Analisis sensitivitas sebagai opsional |
| Split | 80:20, `random_state=42`; A **stratify=y**, B tanpa stratifikasi | Sesuai spesifikasi; stratifikasi wajib dipertimbangkan di A |
| Pemilihan konfigurasi | `GridSearchCV` di training, `StratifiedKFold(5)` (A) / `KFold(5)` (B) | Spesifikasi §4 poin 4 |
| Model A | Baseline Dummy · **SVM (RBF, class_weight balanced)** · **Random Forest (balanced)** | Wajib SVM & RF; baseline diminta checklist §8 dan "perbandingan tiga pendekatan" di Panduan Klasifikasi §8 |
| Model B | Baseline Dummy (rerata) · **Linear Regression** · **Random Forest Regressor (diregularisasi)** | Wajib LR & RFR; baseline untuk membaca R² |
| Model disimpan | A: model dengan macro-F1 CV terbaik (kemungkinan SVM) · B: model dengan MAE CV terbaik (kemungkinan RF) | Aturan pemilihan ditulis **sebelum** melihat test |
| Struktur folder | Mengikuti folder kelompok (pola UTS): `dataset/`, `notebook/`, `models/`, `figures_and_outputs/`, `streamlit/`, `report/`; encoding bersama di `streamlit/vark_utils.py` | Satu sumber encoding untuk notebook dan Streamlit (§7) |

**Temuan paling penting untuk laporan:**
1. **Klasifikasi VARK:** macro-F1 CV ≈ 0,61 untuk SVM dan RF (baseline 0,09). Hampir semua kesalahan terjadi pada responden yang skor tertingginya **berdekatan** (selisih skor teratas dan kedua 0–1), terutama kelas *Seri*. Ini pola yang wajar karena label dihitung dari aturan skor.
2. **Regresi IPK:** tidak ada model yang lebih baik dari baseline rerata secara meyakinkan di framework mana pun (R² CV ≤ 0). Linear Regression biasa **overfit berat** (R² CV −1,64 pada VARK). Uji permutasi di training: p ≈ 0,6. Ini temuan sah: spesifikasi menilai proses dan interpretasi, bukan besar R².

---

## 1. Analisis spesifikasi: wajib, opsional, dan dilarang

### 1.1 Wajib (dinilai di rubrik)

| Kelompok | Proyek A – Klasifikasi | Proyek B – Regresi |
|---|---|---|
| X | Semua item dari **satu** framework | Semua item dari **satu** framework |
| y | Satu label dari aturan skoring panduan | IPK numerik kolom DL |
| Dikeluarkan dari X | ID, IPS, IPK, skor, label, framework lain | ID, IPS, label gaya, skor, framework lain |
| EDA | Missing, duplikasi, invalid, distribusi kelas, kelas minoritas | Missing, distribusi IPK, nilai ekstrem, hubungan fitur–IPK |
| Split | 80:20, `random_state=42`, stratifikasi bila mungkin | 80:20, `random_state=42` |
| Pipeline | Transformasi yang belajar dari data (scaling, imputasi, seleksi fitur) **fit di training saja**; split sama untuk kedua model | sama |
| Model | SVM dan Random Forest Classifier | Linear Regression dan Random Forest Regressor |
| Metrik test | Accuracy, macro-F1, confusion matrix, contoh salah prediksi | MAE, RMSE, R², satu grafik evaluasi, analisis error |
| Visual | Minimal 1 EDA + 1 evaluasi, dengan interpretasi | sama |
| Simpan | `Model_klasifikasi.joblib` (pipeline lengkap) | `Model_regresi.joblib` (pipeline lengkap) |
| Streamlit | Tab Klasifikasi: input sesuai skema, prediksi + nama framework + penjelasan kategori | Tab Regresi: input sesuai skema, IPK prediksi skala benar + disclaimer + alert bila di luar 0–4 |

### 1.2 Opsional (boleh, asal ditandai "pengembangan opsional")
- Tuning parameter (di training), model tambahan (misalnya Logistic Regression, Ridge), analisis sensitivitas straightliner, feature importance, uji permutasi.

### 1.3 Dilarang / jebakan
- Mengimputasi IPK target, membuat kelas IPK tinggi/sedang/rendah.
- Memakai skor penyusun label (`VARK_V`, `VARK_A`, …, `VARK_total`) atau label sebagai fitur tambahan. Ini berlaku juga untuk *feature engineering* yang menjumlahkan item per modalitas.
- Fit scaler/imputer/seleksi fitur sebelum split; memilih parameter berdasarkan hasil test; mengubah label atau aturan cleaning setelah melihat test.
- Menyamakan jawaban VARK kosong dengan empat nol (berlaku juga di input Streamlit).
- Prediksi Streamlit hard-coded atau dari aturan statis; meminta identitas di aplikasi; mempublikasikan data mentah.

### 1.4 Peta rubrik → bagian notebook

| Rubrik | Bobot | Bagian notebook A | Bagian notebook B |
|---|---|---|---|
| A1/B1 Data & X/y | 10% | A1 | B1 |
| A2/B2 Preprocessing | 15% | A2 | B2 |
| A3/B3 EDA | 10% | A3 | B3 |
| A4/B4 Split & pipeline | 15% | A4 | B4 |
| A5/B5 Dua model | 20% | A5 | B5 |
| A6/B6 Evaluasi & error | 20% | A6 | B6 |
| A7/B7 Reproduksibilitas & Streamlit | 10% | A7 + `app.py` | B7 + `app.py` |

---

## 2. Kondisi data dari UTS yang relevan untuk UAS

| Aspek | Nilai | Dampak |
|---|---|---|
| Baris | 192 (194 awal − 2 kosong total) | Data kecil: test set A = 39 baris, B = 35 baris, jadi metrik test berfluktuasi |
| Missing fitur | 0 pada 64 indikator VARK dan 88 item Likert | Tidak perlu imputasi fitur; imputer di pipeline tidak wajib |
| IPK valid | 173 (19 kosong; termasuk 4 IPK = 0 mahasiswa baru yang di-recode ke NaN di UTS C6) | Dataset regresi = 173 baris |
| IPK | median 3,42, SD 0,45, min 0,87, skew −1,63; 8 nilai ekstrem bawah (valid, dipertahankan) | Error terbesar diperkirakan pada IPK rendah |
| `VARK_label` | Kinesthetic 51, Aural 42, Visual 38, Read/Write 31, Seri 30 | Cukup seimbang; Seri = kelas tersulit |
| `FS_ACT_REF_label` | Active 67, Reflective 64, Setara 61 (15 Setara = straightliner) | Seimbang, tetapi straightliner menumpuk di Setara |
| `HM_label` | Seri 64, Pragmatist 43, Reflector 36, Theorist 31, Activist 18 | Didominasi Seri |
| `KOLB_label` | Batas 70, CE_RO 42, CE_AE 32, AC_RO 30, AC_AE 18 | Didominasi Batas |
| Korelasi item–IPK | \|ρ\| maks 0,20–0,23 per framework, 0 signifikan setelah FDR | Ekspektasi R² rendah sudah terlihat di EDA UTS |

---

## 3. Eksplorasi: framework dan model mana yang cocok

Setup semua eksperimen: split 80:20 `random_state=42` (stratify y untuk klasifikasi), lalu **Repeated CV di training** (klasifikasi 5-fold × 3; regresi 5-fold × 4). Skala dan transformasi berada di dalam pipeline. Untuk framework Likert dicoba juga versi tanpa 15 straightliner.

### 3.1 Klasifikasi: macro-F1 CV (training)

| Framework | n train | SVM terbaik | RF terbaik | Model lain terbaik | Dummy |
|---|---|---|---|---|---|
| **VARK** | 153 | **0,614** (RBF, C=10, balanced) | 0,586 (balanced) | KNN 0,566 | 0,085 |
| Felder AR | 153 | 0,834 (linear, C=10, balanced) | 0,692 | LogReg 0,847 | 0,171 |
| Felder AR tanpa SL | 141 | 0,831 | 0,648 | LogReg 0,834 | 0,182 |
| Kolb | 153 | 0,739 (linear) | 0,545 | LogReg 0,705 | 0,107 |
| Kolb tanpa SL | 141 | 0,742 (linear) | 0,589 | – | 0,095 |
| Honey–Mumford | 153 | 0,563 | 0,603 (balanced) | LogReg 0,604 | 0,100 |
| Honey–Mumford tanpa SL | 141 | 0,541 | 0,555 | – | 0,087 |

**Cara membaca hasil ini (penting untuk bagian analisis laporan):**
- **Felder dan Kolb paling mudah ditiru oleh model linear**, karena labelnya adalah *tanda* dari selisih jumlah item (Active − Reflective; AC − CE dan AE − RO). Aturan seperti itu memang berbentuk garis/bidang pemisah linear. Random Forest jauh lebih buruk di sini karena harus memotong-motong ruang 32 dimensi dengan aturan kotak untuk meniru batas miring.
- **VARK dan Honey–Mumford lebih sulit**, karena labelnya adalah *argmax* dari empat jumlah, ditambah kelas Seri ketika ada skor tertinggi yang sama. Daerah Seri sangat tipis (tepat di perbatasan), jadi sulit dipelajari dari 153 contoh. Di VARK, SVM RBF unggul tipis atas SVM linear (0,614 vs 0,580).
- **Membuang straightliner tidak mengubah kesimpulan** untuk Felder dan Kolb, tetapi menurunkan HM.
- Semua model jauh di atas baseline. Model hanya belajar meniru aturan skoring (spesifikasi §2: *"bukan memvalidasi keberadaan atau keunggulan gaya belajar tertentu"*).

**Tuning VARK (GridSearchCV di training, scoring macro-F1, StratifiedKFold 5):**

| Model | Parameter terbaik | `best_score_` (inner CV) | Nested CV (estimasi jujur) |
|---|---|---|---|
| SVM | `kernel='rbf', C=1, gamma='scale', class_weight='balanced'` | 0,667 | 0,613 ± 0,052 |
| RF | `n_estimators=500, max_features='sqrt', min_samples_leaf=2, max_depth=None, class_weight='balanced'` | 0,590 | 0,610 ± 0,095 |

`best_score_` lebih tinggi dari nested CV karena skor terbaik dari banyak kombinasi parameter cenderung terlalu optimistis. Ini poin bagus untuk ditulis di laporan. SVM dan RF hampir seri, tetapi SVM lebih stabil (SD 0,05 vs 0,10).

**Per kelas (CV 5-fold di training, SVM hasil tuning di atas):** Kinesthetic F1 0,84, Visual 0,78, Aural 0,71, Read/Write 0,61, **Seri 0,40**. Pada konfigurasi SVM lain dan RF, F1 Seri hanya 0,26–0,32. Seri konsisten menjadi kelas tersulit.

**Kesalahan menumpuk pada skor yang berdekatan** (CV training, "gap" = skor modalitas tertinggi − skor kedua):

| Gap | 0 (Seri) | 1 | 2 | 3 | ≥4 |
|---|---|---|---|---|---|
| n | 24 | 34 | 23 | 23 | 49 |
| Error SVM | 62% | 44% | 43% | 22% | 2% |
| Error RF | 71% | 56% | 48% | 26% | 4% |

Tabel ini sebaiknya dibuat ulang di notebook A (di test set dan di CV) karena langsung memenuhi rubrik A6: *"hubungkan kesalahan dengan kelas minoritas, fitur, atau kedekatan skor kategori"*.

### 3.2 Regresi IPK: CV di training (n train = 138)

| Framework | Linear Regression (OLS) | RF terbaik | Ridge (eksplorasi) | Dummy rerata |
|---|---|---|---|---|
| **VARK** | MAE 0,488 · R² **−1,641** | MAE 0,330 · R² −0,032 (`min_samples_leaf=5, max_features='sqrt'`) | MAE 0,335 · R² −0,079 | MAE 0,332 · R² −0,060 |
| Felder | MAE 0,411 · R² −0,741 | MAE 0,329 · R² −0,045 | MAE 0,326 · R² −0,055 | MAE 0,332 · R² −0,060 |
| Honey–Mumford | MAE 0,417 · R² −0,940 | MAE 0,331 · R² −0,056 | MAE 0,335 · R² −0,126 | MAE 0,332 · R² −0,060 |
| Kolb | MAE 0,386 · R² −0,548 | MAE 0,333 · R² −0,106 | MAE 0,331 · R² −0,067 | MAE 0,332 · R² −0,060 |

Eksperimen tambahan pada VARK dan Felder: seleksi fitur `SelectKBest` + LR (k = 5, 10, 20) lebih buruk dari baseline. PCA(5) + LR, Lasso, Huber, SVR, KNN, Extra Trees, dan HistGradientBoosting juga tidak lebih baik dari baseline. Hasil RF dengan GridSearch di dalam nested CV: VARK MAE 0,340 (R² −0,095), Felder MAE 0,330 (R² −0,042). **Uji permutasi** (5-fold di training, target diacak): VARK p = 0,60 (Ridge) dan 0,63 (RF); Felder p = 0,15 dan 0,14. Tidak ada yang signifikan.

**Cara membaca hasil ini:**
- **R² baseline di CV negatif (≈ −0,06)** karena rerata tiap fold training sedikit berbeda dari rerata fold validasi. Jadi patokan "model berguna" adalah *mengalahkan Dummy*, bukan R² > 0.
- **Linear Regression overfit** karena ada 64 fitur untuk 138 baris (rasio fitur:data ≈ 1:2) dan indikator VARK saling berkorelasi. Koefisien menjadi tidak stabil, sehingga prediksi pada data baru meleset jauh. Ridge (regularisasi) mengembalikan error ke level baseline. Ini cerita yang bagus untuk B5/B6.
- **Random Forest dengan daun minimal 5–10 sampel** praktis memprediksi nilai dekat rerata, sehingga error-nya setara baseline. RF default (daun 1 sampel) lebih buruk (R² VARK −0,10).
- **Felder sedikit lebih baik dari VARK**, tetapi selisihnya jauh di bawah variasi antar-fold (SD R² 0,07–0,13) dan uji permutasinya tidak signifikan. Memilih framework regresi dari selisih sekecil ini berarti memilih noise. Karena itu VARK tetap disarankan supaya satu skema dengan Proyek A.

### 3.3 Mengapa VARK, dan kapan Felder lebih baik

| Kriteria | VARK | Felder–Silverman (Active–Reflective) |
|---|---|---|
| Macro-F1 CV klasifikasi | ~0,61 (5 kelas) | ~0,83 (3 kelas) |
| Pengaruh straightliner | Tidak ada (format pilihan) | 15 dari 61 Setara adalah straightliner, jadi perlu keputusan tambahan |
| Keseimbangan kelas | Rasio 1,7 | Rasio 1,1 |
| Analisis kesalahan | Kaya: Seri, gap skor, kebingungan Aural ↔ Read/Write | Lebih sederhana: Setara di batas Active/Reflective |
| Input Streamlit | 16 pertanyaan pilihan jamak (checkbox) | 32 slider Likert |
| Kesesuaian dengan regresi | Satu form input untuk dua tab bila B juga VARK | Dua form berbeda bila B memakai VARK |
| Persiapan dari UTS | 64 indikator sudah ada dan terverifikasi | Item sudah ada |

Rubrik menyatakan nilai **tidak** ditentukan oleh besar accuracy/R². Jadi VARK tidak dirugikan oleh F1 yang lebih rendah, asalkan analisisnya bagus. Pilih Felder untuk Proyek A bila kelompok lebih mengutamakan model yang lebih akurat dan input yang lebih sederhana, dan siap menulis keputusan straightliner.

---

## 4. Strategi Proyek A – Klasifikasi VARK

### A0 Setup
- Import, `SEED = 42`, `TEST_SIZE = 0.2`, path relatif (`PROJECT_DIR` seperti notebook UTS) dan `FEATURE_COLS` dari `streamlit/vark_utils.py`, cetak versi Python/sklearn/pandas (bukti reproduksibilitas).

### A1 Data & definisi X/y
- Baca `dataset/processed_UTS/KelpAdam_survey_prepared.csv` (hasil UTS).
- X = 64 kolom `VARK_Q01_V … VARK_Q16_K` (urutan tetap, disimpan di `vark_utils.py`). y = `VARK_label`.
- Tabel "kolom yang dikeluarkan": ID, IPS_*, IPK, 88 item Likert, `VARK_V/A/R/K/total`, semua label lain, flag.
- **Verifikasi label (A1 skor 5):** hitung ulang skor V/A/R/K dan label dari 64 indikator memakai `streamlit/vark_utils.py`, lalu `assert` sama dengan `VARK_label` UTS untuk 192 baris. Tampilkan 3 contoh manual: satu kelas tunggal, satu Seri, satu responden dengan gap 1.
- Verifikasi encoding: encode ulang 16 kolom teks `VARK_Q01…Q16` → 64 indikator dengan fungsi yang sama dengan Streamlit, lalu `assert` identik dengan kolom UTS. Ini bukti bahwa encoding aplikasi = encoding training.

### A2 Preprocessing & justifikasi
- Ringkas cleaning log UTS (194 → 192, alasan, C1–C10) dan rujuk `KelpAdam_cleaning_log.csv`.
- Validasi ulang: semua indikator ∈ {0,1}; setiap skenario minimal 1 opsi dipilih; 0 missing; 0 duplikat baris.
- **Keputusan straightliner:** dipertahankan, dengan alasan flag berasal dari Likert, bukan VARK. Tampilkan bukti bahwa jumlah pilihan VARK mereka bervariasi. Sebut ID 55 (memilih keempat opsi di 16 soal → Seri) sebagai kasus khusus yang tetap valid menurut aturan.
- Tidak ada scaling di tahap ini. Scaling ada di dalam pipeline (A4).

### A3 EDA (fokus ke keputusan modeling)
1. Tabel frekuensi + bar chart `VARK_label` (n dan %). Kelas minoritas = Seri (30, 16%).
2. Distribusi **gap skor** (tertinggi − kedua) per label. Ini menjelaskan mengapa Seri akan sulit.
3. Heatmap % opsi dipilih (16 skenario × 4 modalitas), rata-rata jumlah pilihan per responden.
4. Kalimat penghubung: stratifikasi → kelas Seri di test hanya ±6; metrik utama macro-F1; class_weight dipertimbangkan.

### A4 Split & pipeline
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```
- Tampilkan distribusi kelas train vs test (hasil eksplorasi: train K41/A33/V30/R25/Seri24, test K10/A9/V8/R6/Seri6).
- Pipeline SVM: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=False))])`. Scaler fit hanya di fold training.
- Pipeline RF: `RandomForestClassifier(...)` tanpa scaler (pohon tidak butuh skala). Tulis alasannya.
- Objek CV dipakai bersama: `cv = StratifiedKFold(5, shuffle=True, random_state=42)`.

### A5 Dua model + baseline
- **Baseline:** `DummyClassifier(strategy="most_frequent")`.
- **SVM:** `GridSearchCV`, `scoring="f1_macro"`, grid:
  `kernel ∈ {linear, rbf}`, `C ∈ {0.1, 1, 10, 100}`, `gamma ∈ {scale, 0.001, 0.01, 0.1}` (rbf), `class_weight ∈ {None, balanced}`.
- **RF:** `GridSearchCV`, `n_estimators=500` tetap, `max_depth ∈ {None, 10}`, `min_samples_leaf ∈ {1, 2, 4}`, `max_features ∈ {sqrt, 0.3}`, `class_weight ∈ {None, balanced}`.
- Tabel perbandingan CV: parameter terbaik, mean ± SD macro-F1 dan accuracy, waktu training.
- **Aturan memilih model (ditulis sebelum test):** macro-F1 CV tertinggi. Bila selisih < 1 SD, pilih yang SD-nya lebih kecil / lebih sederhana. Berdasarkan eksplorasi, kemungkinan besar SVM RBF.
- Opsional: nested CV (§3.1) untuk menunjukkan `best_score_` optimistis; Logistic Regression sebagai pembanding tambahan.

### A6 Evaluasi test (sekali, di akhir)
- Untuk **Dummy, SVM, RF**: accuracy, macro-F1, `classification_report` per kelas.
- Confusion matrix SVM dan RF berdampingan (label urut: Visual, Aural, Read/Write, Kinesthetic, Seri).
- **Tabel contoh salah prediksi:** ID, label asli, prediksi SVM, prediksi RF, skor V/A/R/K, gap. Skor ini hanya untuk analisis, bukan fitur.
- **Error vs gap** di test (tabel seperti §3.1). Kesimpulan yang diharapkan: kesalahan terutama pada gap 0–1, yaitu kasus yang memang hampir seri menurut aturan.
- Interpretasi RF: `permutation_importance` di test (opsional). Peringatan: importance menunjukkan item mana yang dipakai model untuk meniru aturan, bukan "item penentu gaya belajar".
- Keterbatasan: test hanya 39 baris (1 salah ≈ 2,6 poin accuracy), label deterministik dari X, Seri adalah aturan operasional.

### A7 Simpan & reproduksibilitas
- `joblib.dump(best_pipeline, "models/Model_klasifikasi.joblib")`, lalu muat ulang dan `assert` prediksi test identik.
- Simpan `models/model_card_klasifikasi.json`: urutan 64 kolom, daftar kelas, parameter, versi sklearn, metrik test, tanggal.
- Ekspor `dataset/processed_UAS/Dataset_klasifikasi.csv` = ID + 64 indikator + `VARK_label`.
- *Restart & Run All* sebelum dikumpulkan.

---

## 5. Strategi Proyek B – Regresi IPK dari VARK

### B1 Data & definisi X/y
- X = 64 indikator VARK (urutan sama dengan A). y = `IPK` (float, skala 0–4).
- Filter `IPK.notna()` → 173 baris. Tabel alasan 19 baris tidak dipakai: IPK kosong di data dosen, serta 4 IPK = 0 pada mahasiswa tanpa IPS sebelumnya (keputusan UTS C6: "belum ada nilai", bukan IPK nol).
- `assert` 0 ≤ IPK ≤ 4; tampilkan tipe dan rentang.
- Kolom yang dikeluarkan: ID, semua IPS (termasuk yang sangat berkorelasi dengan IPK, jadi akan bocor), label, skor, framework lain.

### B2 Preprocessing & justifikasi
- **Target tidak diimputasi** (panduan B §2.4).
- Fitur: 0 missing. Bila ingin aman untuk input aplikasi, `SimpleImputer` boleh ada di pipeline, tetapi tidak diperlukan (lihat trade-off T9).
- **8 IPK ekstrem bawah dipertahankan**: IPK dari sistem, konsisten dengan IPS (selisih maks 0,21 di UTS C7). Tidak ada dasar untuk membuangnya.
- Straightliner dipertahankan (alasan sama dengan A2). Opsional: tabel sensitivitas hasil dengan/tanpa straightliner.

### B3 EDA
1. Histogram + boxplot IPK, statistik (mean 3,36; median 3,42; SD 0,45; skew −1,63), garis Q1 − 1,5 IQR.
2. Missing IPK per kategori (kosong vs recode 0).
3. Hubungan fitur–IPK: bar Spearman ρ untuk 64 indikator (|ρ| maks 0,20, 0 signifikan setelah FDR); boxplot IPK untuk 2–3 indikator teratas (dipilih vs tidak dipilih).
4. Kalimat penghubung: sinyal lemah → ekspektasi R² ≈ 0; RF perlu diregularisasi; OLS berisiko overfit karena 64 fitur vs 138 baris.

### B4 Split & pipeline
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```
- Tampilkan ringkasan IPK train vs test (eksplorasi: rerata 3,36 vs 3,37; SD 0,45 vs 0,45; IPK < 2,5: 6 di train, 2 di test; minimum test 2,24). Distribusi cukup mirip, jadi split tanpa stratifikasi dapat dipertanggungjawabkan.
- Pipeline LR: `Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])`. Prediksi OLS tidak berubah karena scaling, tetapi koefisien jadi dapat dibandingkan dan pipeline seragam.
- Pipeline RF: `RandomForestRegressor(n_estimators=500, random_state=42)`.
- `cv = KFold(5, shuffle=True, random_state=42)` dipakai bersama.

### B5 Dua model + baseline
- **Baseline:** `DummyRegressor(strategy="mean")`.
- **Linear Regression:** tanpa hyperparameter; laporkan MAE/RMSE/R² CV dan jelaskan overfit.
- **RF Regressor:** `GridSearchCV`, `scoring="neg_mean_absolute_error"`, grid `min_samples_leaf ∈ {3, 5, 10, 20}`, `max_features ∈ {sqrt, 0.3}`, `max_depth ∈ {None, 5}`. Eksplorasi memilih `min_samples_leaf=5, max_features='sqrt'`.
- **Aturan memilih model (sebelum test):** MAE CV terendah di antara LR dan RF. Berdasarkan eksplorasi: RF.
- Opsional: Ridge (`RidgeCV`) sebagai "LR dengan regularisasi" untuk menunjukkan bahwa masalah LR adalah varians, bukan bug.
- Opsional: `permutation_test_score` di training untuk menjawab "apakah ada sinyal sama sekali?".

### B6 Evaluasi test
- Tabel MAE, RMSE, R² untuk Dummy, LR, RF. Baca MAE dalam poin IPK dan bandingkan dengan SD IPK 0,45.
- **Grafik 1:** aktual vs prediksi (dua model, garis y = x). Pola yang diharapkan: prediksi RF menumpuk dekat rerata (regression to the mean).
- **Grafik 2:** residual vs IPK aktual. Residual positif besar pada IPK rendah: model menebak terlalu tinggi.
- Tabel 5 error terbesar (ID, IPK aktual, prediksi, residual) dan MAE per kelompok IPK (< 3,0; 3,0–3,5; > 3,5).
- Interpretasi wajib: bukan hubungan sebab-akibat; urutan waktu survei vs IPK tidak diketahui; R² negatif berarti lebih buruk dari menebak rerata.

### B7 Simpan & reproduksibilitas
- `models/Model_regresi.joblib` + muat ulang + `assert` prediksi identik.
- `models/model_card_regresi.json`: kolom, parameter, versi, metrik test, **rentang IPK training** (untuk alert di aplikasi).
- `dataset/processed_UAS/Dataset_regresi.csv` = ID + 64 indikator + IPK.

---

## 6. Trade-off yang perlu diputuskan dan ditulis di laporan

| # | Pilihan | Opsi A | Opsi B | Saran |
|---|---|---|---|---|
| T1 | Framework sama atau beda | Sama (VARK–VARK): satu form input, satu encoder, kode lebih sedikit | Beda (mis. Felder–VARK): F1 klasifikasi lebih tinggi, tetapi dua form | Sama, kecuali kelompok memilih Felder untuk A (§3.3) |
| T2 | Straightliner | Pertahankan: n lebih besar, sesuai prinsip "tidak hapus otomatis" | Buang: kualitas Likert lebih bersih, tetapi tidak relevan untuk VARK | Pertahankan; sensitivitas opsional |
| T3 | Metrik pemilihan A | Accuracy | **Macro-F1**: kelas Seri ikut dihitung setara | Macro-F1 |
| T4 | `class_weight` | None | `balanced`: membantu Seri/Read-Write | Masukkan ke grid; biarkan CV memilih |
| T5 | Tuning | Default param: sederhana, tidak ada risiko overfit pada CV | GridSearchCV: dinilai positif (rubrik "setelan dibandingkan"), tetapi `best_score_` optimistis | GridSearch dengan grid kecil + laporkan nested CV (opsional) |
| T6 | Stratifikasi B | Tanpa stratifikasi: sesuai teks spesifikasi | Stratifikasi bin IPK: test lebih representatif | Tanpa; distribusi train/test sudah mirip |
| T7 | Kernel SVM | Linear: interpretasi mudah | **RBF**: menangkap argmax + Seri lebih baik (0,614 vs 0,580) | Biarkan grid memilih; jelaskan |
| T8 | RF regresi | Default (daun 1): overfit | Daun 5–10, `max_features='sqrt'`: stabil, mendekati rerata | Diregularisasi |
| T9 | Imputer di pipeline | Tanpa: data training lengkap; aplikasi memvalidasi input | Dengan `SimpleImputer`: robust, tetapi menyembunyikan input kosong | Tanpa imputer + validasi wajib-isi di Streamlit (sesuai "jangan ubah kosong jadi nol") |
| T10 | Feature engineering skor | Tambah `VARK_V…K` sebagai fitur: F1 ≈ 100% | – | **Ditolak**: dilarang panduan; sebut di laporan sebagai alasan mengapa skor hanya untuk audit |
| T11 | SMOTE / data sintetis | SMOTE di dalam pipeline | Respons sintetis berlabel aturan | **Ditolak**: imbalance ringan (rasio 1,7), SMOTE menghasilkan nilai non-biner, data sintetis melanggar aturan "jangan mengarang" |
| T12 | Encoding di app | Duplikasi fungsi di `app.py` | **Modul `streamlit/vark_utils.py` bersama** + assert di notebook | Modul bersama |
| T13 | Isi joblib | Pipeline dengan transformer kustom (teks → indikator) | **Pipeline sklearn murni** dengan input 64 indikator; teks → indikator di `streamlit/vark_utils.py` | Sklearn murni: joblib tidak bergantung pada kelas kustom, lebih aman saat deploy |
| T14 | Versi library | Bebas | **Pin `scikit-learn==<versi training>`** di requirements | Pin: file joblib sensitif versi |
| T15 | Ukuran test kecil | Hanya metrik test | Metrik test + CV training sebagai pelengkap | Keduanya, dengan label jelas mana yang test |

---

## 7. Struktur folder (mengikuti folder kelompok, pola UTS)

Folder kerja: `D:\Kuliah\Semester 5\Machine Learning\KelpAdam_UAS_MachineLearning`. Yang bertanda **(baru)** belum ada dan akan dibuat saat pengerjaan.

```
KelpAdam_UAS_MachineLearning/
├── dataset/
│   ├── raw/
│   │   └── Survei_Gaya_Belajar_IPS_IPK_NO_PII.xlsx      # data dosen, read-only, jangan di-commit
│   ├── processed_UTS/
│   │   └── KelpAdam_survey_prepared.csv                 # output UTS (192 × 216) = input kedua notebook
│   └── processed_UAS/                     (baru)        # dataset siap model, output notebook
│       ├── Dataset_klasifikasi.csv                      # ID + 64 indikator + VARK_label (192 baris)
│       └── Dataset_regresi.csv                          # ID + 64 indikator + IPK (173 baris)
│
├── notebook/
│   ├── Notebook_A_Klasifikasi.ipynb       (baru)        # A0–A7
│   └── Notebook_B_Regresi_IPK.ipynb       (baru)        # B0–B7
│
├── models/
│   ├── Model_klasifikasi.joblib           (baru)        # pipeline lengkap (scaler + model)
│   ├── Model_regresi.joblib               (baru)
│   ├── model_card_klasifikasi.json        (baru)        # urutan kolom X, versi sklearn, metrik test
│   └── model_card_regresi.json            (baru)        # + rentang IPK training (untuk alert app)
│
├── figures_and_outputs/                                 # grafik & tabel yang disimpan dari notebook
│   ├── A_*.png, A_*.csv                   (baru)        # mis. A3_distribusi_label.png, A5_cv_results.csv
│   └── B_*.png, B_*.csv                   (baru)        # mis. B6_aktual_vs_prediksi.png, B6_metrik_test.csv
│
├── streamlit/
│   ├── app.py                             (baru)        # 2 tab: Klasifikasi & Regresi
│   └── vark_utils.py                      (baru)        # skema 16 soal VARK + encode jawaban → 64 indikator
│                                                        # + urutan kolom; dipakai app DAN notebook
├── report/
│   └── Laporan_UAS.pdf (+ .docx)          (baru)
│
├── KelpAdam_Strategi_UAS.md                             # dokumen ini
├── README.md                                            # cara install, jalankan notebook & app, tautan deploy
├── requirements.txt                                     # versi dipin (sama dengan .venv saat training)
├── anggota_dan_tautan.txt                               # (ganti nama dari KelpAdam_anggota.txt, lihat 7.2)
└── .gitignore                             (baru)        # dataset/raw/, .venv/, __pycache__/, .ipynb_checkpoints/
```

### 7.1 Fungsi tiap bagian
- **`dataset/processed_UTS/`** adalah satu-satunya sumber data untuk kedua notebook (Panduan: *"UAS menggunakan dataset yang dihasilkan pada UTS"*). File ini tidak diubah; notebook hanya membaca, lalu menulis hasilnya ke `processed_UAS/`.
- **`streamlit/vark_utils.py`** menggantikan `src/` di versi awal dokumen ini. Isinya skema VARK (teks 16 soal + 4 opsi, A=V B=A C=R D=K), fungsi `encode_answers()` → 64 indikator, fungsi skor/label (untuk audit saja), dan `FEATURE_COLS`. Notebook memakainya dengan `sys.path.append(str(PROJECT_DIR / "streamlit"))`, lalu membuktikan dengan `assert` bahwa hasil encoding = 64 kolom UTS. Ini memenuhi syarat *"Pengodean input aplikasi harus konsisten dengan preprocessing saat training"*.
- **`models/`** hanya berisi pipeline sklearn murni (scaler + model) dengan input 64 kolom berurutan. `app.py` memuatnya lewat `Path(__file__).resolve().parent.parent / "models"`, jadi tidak bergantung pada folder tempat perintah dijalankan.
- **`figures_and_outputs/`**: semua grafik dan tabel disimpan dari kode (prefix `A_`/`B_` + nomor bagian), sehingga angka di laporan dapat ditelusuri ke notebook.
- Aplikasi **tidak** membaca folder `dataset/` sama sekali, sehingga dataset tidak wajib ikut ke repo publik.

### 7.2 Catatan penyesuaian
- **Nama berkas inti mengikuti spesifikasi §6 persis**: `Notebook_A_Klasifikasi.ipynb`, `Notebook_B_Regresi_IPK.ipynb`, `Dataset_klasifikasi.csv`, `Dataset_regresi.csv`, `Model_klasifikasi.joblib`, `Model_regresi.joblib`, `app.py`, `requirements.txt`, `README.md`, `Laporan_UAS.pdf`, dan **`anggota_dan_tautan.txt`** (bukan `KelpAdam_anggota.txt` seperti di UTS). Prefix `KelpAdam_` cukup di nama folder/ZIP.
- **Deploy Streamlit Community Cloud**: karena `app.py` ada di subfolder, isi *Main file path* = `streamlit/app.py`. Periksa log deploy bahwa `requirements.txt` di root terbaca; bila tidak, salin juga ke `streamlit/`.
- **`dataset/raw/` di-gitignore** (pelajaran UTS: file mentah sempat ter-commit). Pastikan repo **Private** bila riwayat lama ikut terbawa.

---

## 8. Alur kerja end-to-end

```
 dataset/processed_UTS/KelpAdam_survey_prepared.csv  (UTS, 192 × 216)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Notebook A (VARK_label)        Notebook B (IPK)
  A1 pilih X(64) & y             B1 pilih X(64) & y, filter IPK valid (173)
  A1 assert label & encoding     B2 tanpa imputasi target, outlier dipertahankan
  A2 validasi & keputusan SL     B3 EDA IPK & fitur–IPK
  A3 EDA kelas & gap skor        B4 split 80:20 seed 42
  A4 split 80:20 seed 42 strat.  B5 Dummy · LR · RF (GridSearch, KFold)
  A5 Dummy · SVM · RF            ── pilih model dari CV (test belum dilihat) ──
     (GridSearch, StratKFold)    B6 test: MAE/RMSE/R², grafik, error terbesar
  ── pilih model dari CV ──      B7 simpan + muat ulang + model card
  A6 test: acc, macro-F1, CM,
     contoh salah, error vs gap
  A7 simpan + muat ulang + card
        │                              │
        ▼                              ▼
 models/Model_klasifikasi.joblib   models/Model_regresi.joblib
 dataset/processed_UAS/Dataset_klasifikasi.csv   dataset/processed_UAS/Dataset_regresi.csv
 figures_and_outputs/ (A_*, B_*)
        │                              │
        └──────────────┬───────────────┘
                       ▼
        streamlit/app.py  ← streamlit/vark_utils.py, models/*.joblib
        Tab Klasifikasi | Tab Regresi
                       ▼
        GitHub → Streamlit Community Cloud → tautan di README & anggota_dan_tautan.txt
                       ▼
        Laporan_UAS.pdf (gabungan A & B) → ZIP → LMS (semua anggota upload)
```

**Urutan pengerjaan yang disarankan:**
1. Struktur folder sudah ada. Lengkapi `streamlit/vark_utils.py` (skema, encoding, `FEATURE_COLS`), `.gitignore`, dan `requirements.txt`.
2. Notebook A dan Notebook B dikerjakan paralel (keduanya bergantung pada `streamlit/vark_utils.py`).
3. `app.py` bisa mulai dengan model sementara, lalu diganti model final.
4. Deploy, uji dari browser lain, lalu tulis README dan tautan.
5. Laporan: tarik grafik/tabel dari `figures_and_outputs/`.
6. *Restart & Run All* kedua notebook, cek ulang model termuat di app, lalu susun ZIP.

---

## 9. Desain aplikasi Streamlit

**Input (sama untuk kedua tab, karena satu framework):** 16 pertanyaan VARK, masing-masing berupa 4 checkbox dengan teks opsi persis dari `vark_utils.py`. Aplikasi **menolak memproses** bila ada pertanyaan tanpa pilihan (tampilkan pesan error), karena data training tidak punya skenario kosong dan panduan melarang mengubah kosong menjadi empat nol. Tidak ada input identitas.

Input dapat dibuat sekali dengan `st.session_state`, sehingga pengguna tidak perlu mengisi ulang saat pindah tab. Atau setiap tab memiliki form sendiri dengan `key` berbeda.

**Tab Klasifikasi:**
- `encode_answers()` → DataFrame 1 × 64 dengan urutan `FEATURE_COLS` → `model_klf.predict()`.
- Tampilkan: nama framework (VARK), kategori prediksi, penjelasan singkat kategori (dari `vark_utils.py`), serta catatan bahwa model meniru aturan skoring tugas dan Seri bukan gaya belajar resmi.
- Opsional (dalam expander "Audit"): skor V/A/R/K dari aturan, untuk transparansi. Prediksi tetap dari model. Bila prediksi model berbeda dari aturan, jelaskan bahwa model tidak sempurna (kaitkan dengan gap skor).

**Tab Regresi:**
- Input sama → `model_reg.predict()` → tampilkan IPK dengan 2 desimal.
- Alert bila prediksi < 0 atau > 4 (panduan B §6.3). RF secara alami tidak keluar dari rentang training, tetapi pengecekan tetap ditulis.
- Disclaimer: bukan IPK resmi, bukan jaminan akademik, model tidak lebih baik dari rerata pada data test (tulis MAE test sebagai gambaran ketidakpastian).

**Teknis:** `@st.cache_resource` untuk `joblib.load`; path relatif `Path(__file__).parent / "models"`; cek versi sklearn terhadap model card dan tampilkan peringatan bila berbeda.

---

## 10. Checklist anti-leakage dan kepatuhan spesifikasi

- [ ] X hanya 64 indikator VARK; `assert` tidak ada kolom `VARK_V/A/R/K/total`, `*_label`, `IPS_*`, `IPK` (di A), `ID` di X.
- [ ] Split dibuat **sekali** di awal, sebelum scaler/GridSearch; split sama untuk semua model dalam satu proyek.
- [ ] Scaler berada di dalam `Pipeline`, bukan `scaler.fit(X)` di luar.
- [ ] `GridSearchCV` hanya `.fit(X_train, y_train)`; `X_test` dipakai sekali di A6/B6.
- [ ] Aturan pemilihan model ditulis di markdown sebelum sel evaluasi test.
- [ ] Tidak ada imputasi IPK; 19 baris tanpa IPK dijelaskan.
- [ ] Label tidak diubah setelah melihat hasil.
- [ ] Model yang disimpan = pipeline yang dievaluasi; muat ulang menghasilkan prediksi identik.
- [ ] Streamlit memakai model tersimpan dan `streamlit/vark_utils.py`; tidak ada aturan label hard-coded sebagai output utama.
- [ ] Minimal 1 grafik EDA + 1 grafik evaluasi per proyek, masing-masing dengan interpretasi.
- [ ] Tidak ada data mentah / data pribadi di repo publik.

---

## 11. Usulan pembagian tugas (sesuaikan)

| Anggota | Fokus | Review silang |
|---|---|---|
| Edward | Notebook A + bagian klasifikasi laporan | Notebook B |
| Viriya | Notebook B + bagian regresi laporan | `app.py` |
| Andreas | `streamlit/vark_utils.py`, `app.py`, deploy, README, requirements, `anggota_dan_tautan.txt` | Notebook A |
| Bersama | Ringkasan, keterbatasan, susun ZIP | – |

Spesifikasi menyatakan setiap anggota dapat diminta menjelaskan seluruh pengerjaan, jadi review silang sebaiknya wajib.

---

## Lampiran – Detail eksperimen eksplorasi

**Lingkungan:** Python 3.13, scikit-learn 1.9.1, pandas 3.0.5. Semua `random_state=42`.

**Model yang dicoba (klasifikasi):** Dummy (most frequent), SVM linear (C 0,1/1/10, ± balanced), SVM RBF (default; C 10 balanced; versi tanpa scaling dengan beberapa gamma), SVM poly derajat 2, Random Forest (default, balanced, variasi `min_samples_leaf`/`max_features`), Extra Trees, HistGradientBoosting, Logistic Regression (C 1/10/100), LinearSVC Crammer-Singer, KNN (k = 7). SVM polinomial dan KNN konsisten paling buruk di semua framework.

**Model yang dicoba (regresi):** Dummy (mean, median), Linear Regression, PCA(5) + LR, SelectKBest(k = 5/10/20) + LR, RidgeCV, LassoCV, Huber, Random Forest (default; daun 5 + sqrt; daun 10 + 0,3; kriteria absolute error), Extra Trees, HistGradientBoosting dangkal, SVR RBF, KNN (k = 15), RF dengan GridSearch di nested CV, uji permutasi (Ridge 200 permutasi, RF 50 permutasi).

**Yang sengaja tidak dilakukan:** evaluasi pada test set, karena pemilihan harus berdasarkan training. Angka test pertama kali muncul di notebook kelompok, di bagian A6/B6.
