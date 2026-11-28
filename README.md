# KelpAdam – UAS Pembelajaran Mesin (Gasal 2026/2027)

Proyek UAS kelompok **KelpAdam** dengan framework **VARK**, memakai dataset survei hasil UTS:

- **Proyek A – Klasifikasi:** memprediksi kategori preferensi belajar VARK (Visual, Aural, Read/Write, Kinesthetic, Seri) dengan **SVM** dan **Random Forest Classifier**.
- **Proyek B – Regresi:** memprediksi **IPK** (skala 0–4) dengan **Linear Regression** dan **Random Forest Regressor**.

Keduanya dirangkum dalam satu aplikasi Streamlit dengan dua tab.

| | Tautan |
|---|---|
| Aplikasi Streamlit | `https://<isi-setelah-deploy>.streamlit.app` |
| Repository kode | `https://github.com/<username>/<nama-repo>` |

---

## Ringkasan hasil

| Proyek | Model terpilih | Hasil di data test | Pembanding (baseline) |
|---|---|---|---|
| A – Klasifikasi | SVM (rbf, C=1, gamma=scale, class_weight=balanced) | accuracy 0,641 · macro-F1 0,614 (39 responden) | accuracy 0,256 · macro-F1 0,082 |
| B – Regresi IPK | Random Forest (500 pohon, min_samples_leaf=3, max_features=sqrt, max_depth=5) | MAE 0,345 · RMSE 0,427 · R² 0,07 (35 responden) | MAE 0,356 · R² 0,00 |

- **Klasifikasi:** model jauh di atas baseline. Kesalahan menumpuk pada kelas **Seri** dan pada responden yang selisih skor dua modalitas terkuatnya 0–1.
- **Regresi:** semua model praktis setara dengan menebak rata-rata IPK (uji permutasi p = 0,29; bootstrap CI selisih MAE mencakup 0). Jawaban VARK hampir tidak membawa informasi tentang IPK. Ini temuan, bukan kesalahan pipeline.

Detail ada di notebook, di `TUNING_DAN_EKSPERIMEN.md`, dan di `report/Laporan_UAS.pdf`.

---

## Struktur folder

```
KelpAdam_UAS_MachineLearning/
├── dataset/
│   ├── raw/                        # data mentah dari dosen (TIDAK di-commit, lihat .gitignore)
│   ├── processed_UTS/              # KelpAdam_survey_prepared.csv (192 × 216) = input kedua notebook
│   └── processed_UAS/              # Dataset_klasifikasi.csv, Dataset_regresi.csv (64 fitur + target, tanpa ID)
├── notebook/
│   ├── Notebook_A_Klasifikasi.ipynb
│   └── Notebook_B_Regresi_IPK.ipynb
├── models/
│   ├── Model_klasifikasi.joblib    # pipeline lengkap (StandardScaler + SVM)
│   ├── Model_regresi.joblib        # pipeline lengkap (Random Forest)
│   └── model_card_*.json           # urutan fitur, parameter, metrik, versi library
├── figures_and_outputs/            # semua grafik & tabel yang disimpan otomatis oleh notebook
├── streamlit/
│   └── app.py                      # aplikasi Streamlit satu file (tab Klasifikasi & Regresi IPK)
├── .streamlit/
│   └── config.toml                 # tema tampilan aplikasi (warna & huruf), tidak memengaruhi model
├── report/                         # Laporan_UAS.pdf
├── requirements.txt
├── README.md
├── anggota_dan_tautan.txt
└── TUNING_DAN_EKSPERIMEN.md
```

---

## Cara menjalankan

### 1. Siapkan environment (Python 3.12)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Versi library di `requirements.txt` sengaja dipin sama dengan saat training (scikit-learn 1.9.1), supaya file `.joblib` bisa dimuat ulang dengan hasil yang identik.

### 2. Menjalankan ulang notebook

Buka `notebook/Notebook_A_Klasifikasi.ipynb` atau `notebook/Notebook_B_Regresi_IPK.ipynb`, lalu pilih **Restart & Run All**. Kedua notebook berdiri sendiri: keduanya membaca `dataset/processed_UTS/KelpAdam_survey_prepared.csv` lalu menghasilkan dataset, grafik/tabel, model `.joblib`, dan model card dari awal. Semua proses acak memakai `random_state=42`, jadi angkanya sama setiap kali dijalankan.

### 3. Menjalankan aplikasi Streamlit di laptop

Dari folder utama proyek:

```bash
streamlit run streamlit/app.py
```

Aplikasi terbuka di `http://localhost:8501`. Isi 16 pertanyaan VARK (boleh mencentang lebih dari satu opsi), lalu lihat hasilnya di tab **Klasifikasi VARK** dan **Regresi IPK**.

---

## Deploy ke Streamlit Community Cloud

1. Push repository ke GitHub (folder `models/` dan `streamlit/` wajib ikut).
2. Buka [share.streamlit.io](https://share.streamlit.io), lalu pilih **Create app → Deploy a public app from GitHub**.
3. Isi **Repository** dan **Branch** (`main`), lalu **Main file path** = `streamlit/app.py`.
4. Di **Advanced settings**, pilih **Python 3.12**.
5. Klik **Deploy**. `requirements.txt` di folder utama otomatis dipasang. Setelah selesai, salin URL aplikasi ke tabel tautan di atas dan ke `anggota_dan_tautan.txt`.

---

## Catatan desain aplikasi

- **Output berasal dari model tersimpan.** `app.py` memuat `models/*.joblib` dengan `joblib.load` dan tidak melatih ulang. Tidak ada label atau IPK yang di-*hard-code*.
- **Encoding identik dengan training.** `app.py` memuat teks 16 pertanyaan, `VARK_OPSI`, dan fungsi `encode_vark` yang disalin persis dari Notebook A (sel A2.0) dan Notebook B (sel B2.0). Centang checkbox diubah menjadi teks jawaban berformat sama dengan data survei (beberapa opsi digabung dengan `;`), lalu diproses oleh `encode_vark` tersebut. Urutan 64 kolom dicek terhadap model card setiap kali aplikasi dibuka. Bukti alur yang sama di notebook: sel A7.3 dan B7.3 (teks jawaban → `encode_vark` → model tersimpan menghasilkan prediksi yang sama dengan evaluasi test).
- **Tata letak.** Kuesioner 16 pertanyaan ada di kiri. Panel hasil ada di kanan dan ikut menempel saat halaman digulir. Panel itu berisi tab **Klasifikasi VARK** dan **Regresi IPK**, yang diperbarui otomatis setiap kali jawaban berubah.
- **Jawaban harus lengkap.** Data training tidak punya pertanyaan kosong, jadi aplikasi menolak memprediksi sebelum ke-16 pertanyaan dijawab, bukan mengisinya dengan nol.
- **Tab Regresi** menampilkan IPK numerik dua desimal dan disclaimer bahwa itu bukan IPK resmi. Tab ini memberi peringatan jika prediksi di luar rentang 0–4.
- **Tanpa identitas.** Aplikasi tidak meminta nama, NPM, atau data pribadi lain, tidak membaca dataset, dan tidak menyimpan jawaban pengguna.

## Privasi data

- `dataset/raw/` (file Excel dari dosen) tidak di-commit.
- Dataset di `processed_UTS/` dan `processed_UAS/` sudah tanpa identitas langsung, tetapi tetap berisi data survei mahasiswa. Repository ini dibuat **Private**. Dosen bisa ditambahkan sebagai collaborator.
- Aplikasi Streamlit tidak membutuhkan dataset sama sekali, hanya `models/` dan `streamlit/`.

## Anggota

Nama anggota, pembagian tugas, dan tautan ada di `anggota_dan_tautan.txt`.
