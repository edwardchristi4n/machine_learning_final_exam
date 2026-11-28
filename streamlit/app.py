"""Aplikasi Streamlit KelpAdam - UAS Pembelajaran Mesin (Gasal 2026/2027).

Satu aplikasi, dua tugas, satu framework (VARK):
- Tab Klasifikasi : memprediksi kategori preferensi belajar VARK (model: models/Model_klasifikasi.joblib)
- Tab Regresi IPK : memprediksi IPK numerik skala 0-4 (model: models/Model_regresi.joblib)

Semua output berasal dari model tersimpan yang dimuat ulang (tanpa training ulang), bukan aturan
hard-coded. Pengodean jawaban memakai `encode_vark` yang disalin persis dari notebook, sehingga
identik dengan preprocessing saat training. Aplikasi tidak meminta identitas dan tidak menyimpan jawaban.

Jalankan dari folder utama proyek:
    streamlit run streamlit/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent   # path dihitung dari lokasi file, bukan dari folder tempat perintah dijalankan
MODEL_DIR = PROJECT_DIR / "models"

st.set_page_config(page_title="Prediksi Preferensi Belajar VARK - KelpAdam", page_icon="📘", layout="wide")


# ----------------------------------------------------------------------------------------------
# 0. Skema VARK & encoding (SAMA dengan Notebook A sel A2.0 dan Notebook B sel B2.0)
# ----------------------------------------------------------------------------------------------
FRAMEWORK = "VARK"
MODALITAS = ["V", "A", "R", "K"]   # urutan opsi A, B, C, D pada instrumen
NAMA_MODALITAS = {"V": "Visual", "A": "Aural", "R": "Read/Write", "K": "Kinesthetic"}
FEATURE_COLS = [f"VARK_Q{q:02d}_{m}" for q in range(1, 17) for m in MODALITAS]

# Kalimat pembuka tiap skenario (dari instrumen survei)
VARK_PERTANYAAN = {
    1: "Ketika mempelajari topik baru yang cukup kompleks, saya paling terbantu jika…",
    2: "Saat harus memahami prosedur penggunaan suatu perangkat atau aplikasi, saya cenderung…",
    3: "Untuk mengingat materi menjelang ujian, saya biasanya lebih suka…",
    4: "Ketika dosen menjelaskan konsep abstrak, saya lebih mudah memahami bila…",
    5: "Saat menerima tugas baru, saya biasanya ingin terlebih dahulu…",
    6: "Jika saya tersesat di tempat yang belum dikenal, saya lebih memilih…",
    7: "Saat mempelajari data atau hasil penelitian, saya paling terbantu oleh…",
    8: "Jika harus menjelaskan ide kepada teman, saya cenderung…",
    9: "Dalam kelas daring, saya paling tertarik ketika materi menyediakan…",
    10: "Ketika menghadapi masalah yang belum pernah saya temui, saya biasanya…",
    11: "Saat memilih sumber belajar tambahan, saya lebih tertarik pada…",
    12: "Ketika ingin memastikan saya benar-benar memahami suatu konsep, saya…",
    13: "Jika harus belajar mandiri dari materi yang panjang, saya cenderung…",
    14: "Ketika menerima umpan balik atas tugas, saya lebih terbantu bila…",
    15: "Dalam kerja kelompok, saya biasanya berkontribusi paling baik ketika…",
    16: "Jika harus memilih satu cara untuk mempelajari sesuatu yang benar-benar baru, saya akan memprioritaskan…",
}

# Teks opsi resmi per skenario, urutan [A=Visual, B=Aural, C=Read/Write, D=Kinesthetic] (Lampiran Panduan)
# (disalin apa adanya dari Notebook A sel A2.0 / Notebook B sel B2.0)
VARK_OPSI = {
    1: ["melihat diagram, bagan, peta konsep, atau ilustrasi hubungan antarbagian.",
        "mendengar penjelasan dan dapat berdiskusi atau bertanya secara lisan.",
        "membaca uraian tertulis, definisi, atau catatan yang terstruktur.",
        "mencoba contoh, simulasi, praktik, atau melihat penerapan nyata."],
    2: ["mengikuti gambar langkah demi langkah atau tangkapan layar.",
        "meminta seseorang menjelaskan langkah-langkahnya secara lisan.",
        "membaca petunjuk tertulis dan daftar langkah.",
        "langsung mencoba sambil memperbaiki kesalahan yang muncul."],
    3: ["membuat peta konsep, skema, warna, atau diagram.",
        "menjelaskan materi dengan suara keras atau berdiskusi dengan teman.",
        "menulis ulang ringkasan, membuat daftar, atau membaca catatan.",
        "berlatih soal, mencoba contoh, atau menghubungkan materi dengan pengalaman."],
    4: ["ada visualisasi yang menunjukkan hubungan antar konsep.",
        "penjelasan disampaikan dengan contoh verbal dan tanya jawab.",
        "tersedia teks atau rumusan tertulis yang dapat saya baca kembali.",
        "ada demonstrasi, studi kasus, atau aktivitas langsung."],
    5: ["melihat contoh hasil akhir atau struktur visual tugas.",
        "mendengarkan penjelasan dosen tentang apa yang harus dilakukan.",
        "membaca instruksi dan rubrik secara rinci.",
        "mencoba bagian awal tugas untuk memahami apa yang diminta."],
    6: ["melihat peta atau petunjuk arah visual.",
        "bertanya kepada orang lain dan mendengarkan arahannya.",
        "membaca nama jalan, papan petunjuk, atau instruksi tertulis.",
        "berjalan mencoba rute sambil menyesuaikan arah."],
    7: ["grafik, diagram, dan visualisasi pola.",
        "penjelasan lisan tentang makna hasil.",
        "tabel, uraian tertulis, dan penjelasan metodologis.",
        "contoh penerapan data pada kasus nyata."],
    8: ["menggambar skema atau menunjukkan visual.",
        "menjelaskannya melalui percakapan.",
        "menuliskan poin-poin atau memberi bacaan singkat.",
        "menunjukkan bagaimana ide tersebut bekerja melalui contoh atau demonstrasi."],
    9: ["infografik, diagram, atau video dengan visual yang jelas.",
        "rekaman audio, diskusi sinkron, atau penjelasan verbal.",
        "modul teks, transkrip, dan catatan yang lengkap.",
        "simulasi, kuis interaktif, latihan, atau aktivitas berbasis kasus."],
    10: ["membuat sketsa atau memetakan masalah secara visual.",
         "membicarakan masalah dengan orang lain untuk memperoleh ide.",
         "menuliskan fakta, alternatif, dan langkah pemecahan.",
         "mencoba beberapa solusi dan belajar dari hasil percobaan."],
    11: ["video animasi, diagram interaktif, atau infografik.",
         "podcast, rekaman kuliah, atau diskusi kelompok.",
         "artikel, buku, modul, atau catatan tertulis.",
         "tutorial praktik, proyek, eksperimen, atau studi kasus."],
    12: ["dapat menggambarkan hubungan konsep tersebut dalam bentuk visual.",
         "dapat menjelaskannya secara lisan dengan kata-kata saya sendiri.",
         "dapat menuliskan definisi dan penjelasannya secara runtut.",
         "dapat menerapkannya untuk menyelesaikan masalah atau tugas nyata."],
    13: ["mencari bagian visual terlebih dahulu untuk memahami gambaran besar.",
         "mendengarkan penjelasan terkait sebelum atau sambil belajar.",
         "membaca dan membuat catatan secara sistematis.",
         "mencari latihan atau contoh penerapan setelah memahami pokok materi."],
    14: ["ditunjukkan bagian mana yang perlu diperbaiki secara visual atau dengan markup.",
         "dibicarakan secara langsung sehingga saya dapat bertanya.",
         "diberikan komentar tertulis yang rinci.",
         "diberi kesempatan mencoba revisi dan melihat dampaknya."],
    15: ["menyusun bagan, desain, atau struktur visual.",
         "berdiskusi, mempresentasikan, atau mengoordinasikan secara lisan.",
         "menulis dokumen, rangkuman, atau notulen.",
         "mengerjakan prototipe, praktik, atau implementasi."],
    16: ["melihat representasi visual tentang bagaimana hal itu bekerja.",
         "mendengar seseorang menjelaskannya dan berdialog.",
         "membaca penjelasan tertulis yang lengkap.",
         "mengalami atau mencoba secara langsung."],
}
assert len(VARK_OPSI) == 16 and all(len(v) == 4 for v in VARK_OPSI.values())
assert set(VARK_PERTANYAAN) == set(VARK_OPSI)


def normalisasi(teks):
    """Samakan format teks: huruf kecil, spasi tunggal, tanpa titik di akhir."""
    return " ".join(str(teks).lower().split()).rstrip(".")


# Peta teks opsi -> modalitas, per skenario
PETA_OPSI = {q: {normalisasi(t): m for t, m in zip(opsi, MODALITAS)} for q, opsi in VARK_OPSI.items()}


def encode_vark(jawaban):
    """Ubah 16 jawaban teks VARK menjadi 64 indikator 0/1 (fungsi yang sama dengan notebook).

    jawaban: list/Series berisi 16 teks (skenario 1-16); satu teks bisa memuat beberapa opsi dipisah ';'.
    Mengembalikan (indikator, tidak_cocok):
      - indikator  : dict 64 kolom; skenario kosong -> NaN (bukan empat nol)
      - tidak_cocok: list (skenario, token) yang tidak ada di daftar opsi resmi
    """
    indikator, tidak_cocok = {}, []
    for q, teks in zip(range(1, 17), jawaban):
        token = [] if pd.isna(teks) else [t for t in str(teks).split(";") if t.strip()]  # buang token kosong
        dipilih = set()
        for t in token:
            m = PETA_OPSI[q].get(normalisasi(t))
            if m is None:
                tidak_cocok.append((q, t.strip()))
            else:
                dipilih.add(m)
        for m in MODALITAS:
            indikator[f"VARK_Q{q:02d}_{m}"] = (1 if m in dipilih else 0) if token else np.nan
    return indikator, tidak_cocok


def pilihan_ke_teks(pilihan):
    """Ubah pilihan checkbox aplikasi menjadi teks jawaban berformat sama dengan data survei.

    pilihan: dict {nomor skenario: [teks opsi yang dicentang]}.
    Beberapa opsi digabung dengan ';' persis seperti sel jawaban di dataset, lalu
    hasilnya diproses oleh `encode_vark` yang sama dengan notebook.
    Skenario tanpa centang -> None (dianggap kosong, bukan empat nol).
    """
    return [";".join(pilihan.get(q, [])) or None for q in range(1, 17)]


def jawaban_ke_fitur(pilihan):
    """Pilihan checkbox -> DataFrame 1 x 64 dengan urutan FEATURE_COLS, siap untuk model.predict()."""
    indikator, tidak_cocok = encode_vark(pilihan_ke_teks(pilihan))
    if tidak_cocok:
        raise ValueError(f"Teks opsi tidak dikenali: {tidak_cocok}")
    x = pd.DataFrame([indikator])[FEATURE_COLS]
    if x.isna().any().any():
        raise ValueError("Masih ada skenario yang belum dijawab.")
    return x.astype(int)


def jumlah_pilihan_per_modalitas(x):
    """Jumlah opsi yang dicentang per modalitas (V/A/R/K) - hanya untuk ringkasan, bukan output model."""
    return {NAMA_MODALITAS[m]: int(x[[f"VARK_Q{q:02d}_{m}" for q in range(1, 17)]].sum(axis=1).iloc[0])
            for m in MODALITAS}


# Penjelasan kategori (sama dengan Notebook A sel A7.3 dan model card klasifikasi)
PENJELASAN_KATEGORI = {
    "Visual": "Cenderung paling terbantu oleh representasi visual: diagram, bagan, peta konsep, atau grafik.",
    "Aural": "Cenderung paling terbantu oleh mendengar dan berdiskusi: penjelasan lisan, tanya jawab, atau diskusi kelompok.",
    "Read/Write": "Cenderung paling terbantu oleh teks tertulis: membaca modul, membuat catatan, atau menulis ringkasan.",
    "Kinesthetic": "Cenderung paling terbantu oleh praktik langsung: mencoba contoh, simulasi, latihan, atau studi kasus.",
    "Seri": ("Dua atau lebih modalitas sama kuat. Ini kategori teknis tugas (aturan operasional), "
             "bukan gaya belajar kelima."),
}


def cek_rentang_ipk(nilai):
    """Kembalikan peringatan jika prediksi berada di luar skala IPK 0-4 (sama dengan Notebook B sel B7.3)."""
    if nilai < 0 or nilai > 4:
        return "PERINGATAN: prediksi di luar rentang IPK 0-4"
    return "dalam rentang 0-4"


# ----------------------------------------------------------------------------------------------
# 1. Memuat model tersimpan (sekali saja, lalu disimpan di cache)
# ----------------------------------------------------------------------------------------------
@st.cache_resource
def muat_model(nama_file, nama_card):
    """Muat pipeline .joblib + model card, lalu pastikan urutan fitur sama dengan saat training."""
    model = joblib.load(MODEL_DIR / nama_file)
    card = json.loads((MODEL_DIR / nama_card).read_text(encoding="utf-8"))
    if card["fitur"] != FEATURE_COLS:
        raise RuntimeError(f"Urutan fitur di {nama_card} tidak sama dengan FEATURE_COLS aplikasi.")
    return model, card


try:
    model_klf, card_klf = muat_model("Model_klasifikasi.joblib", "model_card_klasifikasi.json")
    model_reg, card_reg = muat_model("Model_regresi.joblib", "model_card_regresi.json")
except Exception as err:  # file hilang / versi library tidak cocok
    st.error(f"Model gagal dimuat: {err}")
    st.stop()


# ----------------------------------------------------------------------------------------------
# 2. Tampilan: warna, huruf, dan ikon (murni kosmetik, tidak memengaruhi prediksi)
# ----------------------------------------------------------------------------------------------
WARNA_KATEGORI = {"Visual": "#2F6FED", "Aural": "#8B5CF6", "Read/Write": "#1E9E5A",
                  "Kinesthetic": "#F26B1D", "Seri": "#64748B"}
INISIAL_KATEGORI = {"Visual": "V", "Aural": "A", "Read/Write": "R", "Kinesthetic": "K", "Seri": "="}


def ikon(nama, ukuran=18):
    """Ikon garis sederhana (SVG inline) supaya aplikasi tetap satu file tanpa aset eksternal."""
    jalur = {
        "buku": '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/>',
        "daftar": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>',
        "cek": '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.7 2.7L16.5 9.5"/>',
        "seru": '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5v.01"/>',
        "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
        "gembok": '<rect x="4.5" y="10.5" width="15" height="10" rx="2"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>',
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.5v.01"/>',
        "topi": '<path d="m2 9 10-5 10 5-10 5z"/><path d="M6 11v5c3 2.5 9 2.5 12 0v-5"/>',
    }[nama]
    return (f'<svg width="{ukuran}" height="{ukuran}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{jalur}</svg>')


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{
  --navy:#15233B; --navy-2:#22365A; --oranye:#F26B1D; --oranye-tombol:#D4520C; --oranye-muda:#FFF1E8;
  --hijau:#168048; --hijau-muda:#E8F6EE; --kuning:#F5A524; --kuning-muda:#FFF6E0;
  --merah:#D64545; --merah-muda:#FDECEC;
  --latar:#F3F5F9; --kartu:#FFFFFF; --garis:#E3E8F0; --teks:#15233B; --redup:#5B6B82;
}
html, body, .stApp, .stApp p, .stApp div, .stApp span:not([data-testid="stIconMaterial"]), .stApp label, .stApp button, .stApp input { font-family:'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp { background:var(--latar); color:var(--teks); }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display:none; }
.block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1320px; }
h1, h2, h3, h4 { color:var(--teks); letter-spacing:-0.01em; }

/* ---------- sidebar navy ---------- */
section[data-testid="stSidebar"] { background:var(--navy); }
section[data-testid="stSidebar"] * { color:#DCE3EE; }
section[data-testid="stSidebar"] .stMarkdown p { font-size:0.88rem; line-height:1.55; }
.brand { display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.6rem; }
.brand-logo { width:42px; height:42px; border-radius:11px; background:var(--oranye); display:grid; place-items:center; }
.brand-logo svg { color:#fff !important; }
.brand-nama { font-weight:800; font-size:1.25rem; color:#fff !important; line-height:1.1; }
.brand-sub { font-size:.74rem; color:#9FB0C8 !important; }
.side-blok { background:var(--navy-2); border-radius:12px; padding:.9rem 1rem; margin-bottom:.9rem; }
.side-judul { font-weight:700; color:#fff !important; font-size:.9rem; margin-bottom:.55rem; display:flex; gap:.45rem; align-items:center; }
.langkah { display:flex; gap:.6rem; align-items:flex-start; margin:.45rem 0; font-size:.84rem; line-height:1.45; }
.langkah-no { flex:none; width:22px; height:22px; border-radius:50%; background:var(--oranye); color:#fff !important;
  font-weight:700; font-size:.75rem; display:grid; place-items:center; }
.side-penutup { margin-top:1.6rem; font-weight:700; color:#fff !important; font-size:.95rem; line-height:1.35; }
.side-garis { width:56px; height:3px; border-radius:3px; background:var(--oranye); margin-top:.6rem; }

/* ---------- judul halaman ---------- */
.judul-halaman { font-size:2.05rem; font-weight:800; margin:0; line-height:1.15; color:var(--teks); }
.sub-halaman { color:var(--redup); margin:.35rem 0 0; font-size:1rem; }

/* ---------- kartu statistik ---------- */
.grid-stat { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:1rem; }
@media (max-width: 900px){ .grid-stat{ grid-template-columns:repeat(2, minmax(0,1fr)); gap:.7rem; } }
.stat { background:var(--kartu); border:1px solid var(--garis); border-radius:14px; padding:1rem 1.1rem; min-height:118px; }
.stat-atas { display:flex; justify-content:space-between; align-items:flex-start; }
.stat-angka { font-size:1.9rem; font-weight:800; line-height:1; color:var(--teks); }
.stat-label { color:var(--redup); font-size:.86rem; margin-top:.35rem; }
.stat-ikon { width:34px; height:34px; border-radius:50%; display:grid; place-items:center; }
.bar { height:6px; background:#EDF0F5; border-radius:6px; overflow:hidden; margin-top:.75rem; }
.bar > span { display:block; height:100%; border-radius:6px; }
.bar-baris { display:flex; align-items:center; gap:.6rem; }
.bar-baris .bar { flex:1; margin-top:.75rem; }
.bar-persen { font-size:.75rem; color:var(--redup); margin-top:.75rem; min-width:2.4rem; text-align:right; }

/* ---------- kartu pertanyaan ---------- */
div[class*="st-key-kartu_q"] { background:var(--kartu); border-color:var(--garis) !important; border-radius:14px !important; }
.q-kepala { display:flex; gap:.8rem; align-items:flex-start; margin-bottom:.15rem; }
.q-no { flex:none; width:36px; height:36px; border-radius:10px; display:grid; place-items:center;
  font-weight:800; font-size:.9rem; color:#fff; background:var(--navy); }
.q-no.selesai { background:var(--hijau); }
.q-teks { font-weight:700; font-size:.98rem; line-height:1.4; color:var(--teks); padding-top:.45rem; flex:1; }
.chip { display:inline-flex; align-items:center; gap:.3rem; font-size:.72rem; font-weight:700; padding:.2rem .6rem;
  border-radius:999px; white-space:nowrap; }
.chip-hijau { background:var(--hijau-muda); color:var(--hijau); }
.chip-abu { background:#EEF1F5; color:var(--redup); }
.chip-navy { background:#E7ECF5; color:var(--navy); }
.q-chip { padding-top:.5rem; }
@media (max-width: 640px){ .q-kepala{ flex-wrap:wrap; } .q-chip{ width:100%; padding:0 0 0 calc(36px + .8rem); } }

/* opsi jawaban = baris yang bisa diklik */
div[class*="st-key-vark_q"] { width:100%; }
div[class*="st-key-vark_q"] .stCheckbox label { width:100%; border:1px solid var(--garis); border-radius:10px;
  padding:.6rem .8rem; background:#FBFCFE; transition:border-color .15s, background .15s; }
div[class*="st-key-vark_q"] .stCheckbox label:hover { border-color:#F7B48C; }
div[class*="st-key-vark_q"] .stCheckbox label:has(input:checked) { border-color:var(--oranye); background:var(--oranye-muda); }
div[class*="st-key-vark_q"] .stCheckbox label:has(input:checked) > span + div { background:var(--oranye) !important; border-color:var(--oranye) !important; }
div[class*="st-key-vark_q"] .stCheckbox label:has(input:focus-visible) { outline:2px solid var(--oranye); outline-offset:2px; }
div[class*="st-key-vark_q"] .stCheckbox p { font-size:.92rem; color:var(--teks); }

/* ---------- panel hasil (kanan, ikut menempel saat scroll) ---------- */
div[data-testid="stColumn"]:has(.st-key-panel_hasil) { position:sticky; top:3.6rem; align-self:flex-start; }
@media (max-width: 900px){ div[data-testid="stColumn"]:has(.st-key-panel_hasil){ position:static; } }
.st-key-panel_hasil { background:var(--kartu); border:1px solid var(--garis); border-radius:16px; padding:1.1rem 1.2rem 1.2rem;
  max-height:calc(100vh - 4.6rem); overflow-y:auto; scrollbar-width:thin; }
@media (max-width: 900px){ .st-key-panel_hasil{ max-height:none; overflow:visible; } }
.panel-judul { font-weight:800; font-size:1.15rem; margin:0 0 .15rem; color:var(--teks); }
.panel-sub { color:var(--redup); font-size:.85rem; margin:0; }
.st-key-panel_hasil [data-testid="stTab"] p { font-weight:700; color:var(--redup); }
.st-key-panel_hasil [data-testid="stTab"][aria-selected="true"] p { color:var(--oranye); }
.st-key-panel_hasil .react-aria-SelectionIndicator { background:var(--oranye) !important; }

.kosong { text-align:center; padding:1.6rem 1rem 1.2rem; }
.kosong-ikon { width:52px; height:52px; border-radius:50%; background:var(--oranye-muda); color:var(--oranye);
  display:grid; place-items:center; margin:0 auto .8rem; }
.kosong h4 { margin:0 0 .35rem; font-size:1.02rem; }
.kosong p { color:var(--redup); font-size:.87rem; margin:0; line-height:1.5; }

.hasil { border:1px solid var(--garis); border-radius:14px; padding:1rem; margin-top:.3rem; }
.hasil-atas { display:flex; gap:.9rem; align-items:center; }
.hasil-chip { margin-bottom:.75rem; }
.avatar { flex:none; width:52px; height:52px; border-radius:13px; display:grid; place-items:center;
  color:#fff; font-weight:800; font-size:1.35rem; }
.hasil-kecil { color:var(--redup); font-size:.8rem; }
.hasil-besar { font-weight:800; font-size:1.6rem; line-height:1.15; }
.hasil-teks { margin:.85rem 0 0; font-size:.9rem; line-height:1.55; color:var(--teks); }
.catatan { background:#F6F8FB; border-radius:10px; padding:.65rem .8rem; font-size:.82rem; color:var(--redup); line-height:1.5; margin-top:.8rem; }

.sub-judul { font-weight:700; font-size:.92rem; margin:1.1rem 0 .5rem; color:var(--teks); }
.profil-baris { display:grid; grid-template-columns:92px 1fr 44px; align-items:center; gap:.6rem; margin:.4rem 0; font-size:.84rem; }
.profil-baris .bar { margin-top:0; height:8px; }
.profil-nilai { text-align:right; color:var(--redup); font-variant-numeric:tabular-nums; }

.metrik-mini { display:grid; grid-template-columns:1fr 1fr; gap:.6rem; margin-top:.9rem; }
.metrik-mini div { background:#F6F8FB; border-radius:10px; padding:.6rem .75rem; }
.metrik-mini b { display:block; font-size:1.1rem; color:var(--teks); }
.metrik-mini span { font-size:.76rem; color:var(--redup); line-height:1.35; display:block; }

.banner { display:flex; gap:.6rem; align-items:flex-start; border-radius:12px; padding:.75rem .9rem; font-size:.85rem; line-height:1.5; }
.banner-kuning { background:var(--kuning-muda); color:#7A5200; }
.banner-merah { background:var(--merah-muda); color:#8E2323; }
.banner svg { flex:none; margin-top:.1rem; }

.ipk-angka { font-size:3rem; font-weight:800; line-height:1; color:var(--navy); font-variant-numeric:tabular-nums; }
.ipk-skala { color:var(--redup); font-size:1rem; font-weight:600; }
.skala { position:relative; height:10px; border-radius:10px; margin:2.1rem 0 .4rem;
  background:linear-gradient(90deg,#FDE3D3 0%,#FDF0E2 55%,#E3F3EA 100%); }
.skala-titik { position:absolute; top:50%; width:18px; height:18px; border-radius:50%; background:var(--oranye);
  border:3px solid #fff; box-shadow:0 0 0 1px var(--oranye); transform:translate(-50%,-50%); }
.skala-label { position:absolute; bottom:16px; transform:translateX(-50%); font-size:.72rem; font-weight:700;
  color:var(--oranye); white-space:nowrap; }
.skala-rata { position:absolute; top:-5px; width:2px; height:20px; background:var(--navy); transform:translateX(-50%); }
.skala-angka { display:flex; justify-content:space-between; font-size:.72rem; color:var(--redup); }
.legenda { display:flex; flex-wrap:wrap; gap:.3rem 1rem; font-size:.75rem; color:var(--redup); margin-top:.55rem; }
.legenda i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.3rem; vertical-align:-1px; }

.tabel-model { width:100%; border-collapse:collapse; font-size:.82rem; }
.tabel-model td { padding:.45rem .2rem; border-bottom:1px solid var(--garis); vertical-align:top; }
.tabel-model td:first-child { color:var(--redup); width:38%; }
.tabel-model, .tabel-model tr, .tabel-model td { border:none !important; background:transparent !important; }
.tabel-model tr:not(:last-child) td { border-bottom:1px solid var(--garis) !important; }

/* tombol */
button[data-testid="stBaseButton-primary"] { background:var(--oranye-tombol); border-color:var(--oranye-tombol); color:#fff; font-weight:700; }
button[data-testid="stBaseButton-primary"]:hover { background:#B8460A; border-color:#B8460A; }
button[data-testid="stBaseButton-secondary"] { font-weight:600; border-color:var(--garis); background:#fff; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------
# 3. State jawaban
# ----------------------------------------------------------------------------------------------
def kunci(q, i):
    """Key session_state untuk checkbox opsi ke-i pada pertanyaan q."""
    return f"vark_q{q:02d}_{i}"


def kosongkan_jawaban():
    for q in VARK_OPSI:
        for i in range(4):
            st.session_state[kunci(q, i)] = False


def isi_contoh_acak():
    """Isi jawaban acak (1-2 opsi per pertanyaan) untuk uji coba; tidak memakai data responden mana pun."""
    rng = np.random.default_rng()
    for q in VARK_OPSI:
        dipilih = set(rng.choice(4, size=rng.integers(1, 3), replace=False).tolist())
        for i in range(4):
            st.session_state[kunci(q, i)] = i in dipilih


pilihan = {q: [opsi for i, opsi in enumerate(VARK_OPSI[q]) if st.session_state.get(kunci(q, i), False)]
           for q in VARK_OPSI}
belum_dijawab = [q for q, opsi in pilihan.items() if not opsi]
n_total = len(VARK_OPSI)
n_terjawab = n_total - len(belum_dijawab)
persen = round(100 * n_terjawab / n_total)

x_input = None
if not belum_dijawab:
    x_input = jawaban_ke_fitur(pilihan)   # teks opsi -> encode_vark (sama dengan notebook) -> DataFrame 1 x 64


# ----------------------------------------------------------------------------------------------
# 4. Sidebar
# ----------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-logo">{ikon("buku", 22)}</div>'
        '<div><div class="brand-nama">KelpAdam</div><div class="brand-sub">Prediksi preferensi belajar</div></div></div>'
        f'<div class="side-blok"><div class="side-judul">{ikon("daftar", 16)} Cara memakai</div>'
        '<div class="langkah"><span class="langkah-no">1</span><span>Jawab 16 pertanyaan. Boleh mencentang lebih dari satu opsi.</span></div>'
        '<div class="langkah"><span class="langkah-no">2</span><span>Hasil muncul di panel kanan setelah semua pertanyaan terjawab.</span></div>'
        '<div class="langkah"><span class="langkah-no">3</span><span>Pindah tab untuk melihat kategori VARK atau prediksi IPK.</span></div></div>'
        f'<div class="side-blok"><div class="side-judul">{ikon("gembok", 16)} Privasi</div>'
        '<p style="margin:0">Aplikasi tidak meminta nama, NPM, atau identitas lain, dan jawabanmu tidak disimpan.</p></div>'
        f'<div class="side-blok"><div class="side-judul">{ikon("topi", 16)} Tentang proyek</div>'
        f'<p style="margin:0">UAS Pembelajaran Mesin, Gasal 2026/2027. Framework yang dipakai: <b>{FRAMEWORK}</b> '
        '(Visual, Aural, Read/Write, Kinesthetic).</p></div>'
        '<div class="side-penutup">Gambaran dari model,<br>bukan penilaian resmi.</div><div class="side-garis"></div>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------
# 5. Judul + kartu statistik
# ----------------------------------------------------------------------------------------------
kol_judul, kol_aksi = st.columns([3, 1.25], vertical_alignment="bottom")
with kol_judul:
    st.markdown('<div class="judul-halaman">Prediksi Preferensi Belajar</div>'
                '<div class="sub-halaman">Isi kuesioner VARK, lalu lihat kategori gaya belajar dan perkiraan IPK '
                'dari model yang sudah dilatih.</div>', unsafe_allow_html=True)
with kol_aksi:
    kol_b1, kol_b2 = st.columns(2)
    kol_b1.button("Isi contoh acak", on_click=isi_contoh_acak, type="primary", width="stretch",
                  help="Mengisi jawaban acak untuk uji coba. Tidak memakai data responden mana pun.")
    kol_b2.button("Kosongkan", on_click=kosongkan_jawaban, width="stretch")

st.write("")
kartu_stat = [
    (str(n_total), "Pertanyaan VARK", "daftar", "#E7ECF5", "var(--navy)", None),
    (str(n_terjawab), "Sudah dijawab", "cek", "var(--hijau-muda)", "var(--hijau)", ("var(--hijau)", persen)),
    (str(len(belum_dijawab)), "Belum dijawab", "seru", "var(--oranye-muda)", "var(--oranye)", ("var(--oranye)", 100 - persen)),
    ("2", "Model dimuat", "chip", "#EEF1F5", "var(--redup)", None),
]
html_stat = []
for angka, label, nama_ikon, latar_ikon, warna_ikon, bar in kartu_stat:
    if bar:
        isi_bar = (f'<div class="bar-baris"><div class="bar"><span style="width:{bar[1]}%;background:{bar[0]}"></span></div>'
                   f'<span class="bar-persen">{bar[1]}%</span></div>')
    else:
        keterangan = "dari file .joblib" if nama_ikon == "chip" else "4 opsi per pertanyaan"
        isi_bar = f'<div class="stat-label" style="margin-top:.75rem">{keterangan}</div>'
    html_stat.append(f'<div class="stat"><div class="stat-atas"><div><div class="stat-angka">{angka}</div>'
                     f'<div class="stat-label">{label}</div></div>'
                     f'<div class="stat-ikon" style="background:{latar_ikon};color:{warna_ikon}">{ikon(nama_ikon)}</div></div>'
                     f'{isi_bar}</div>')
st.markdown(f'<div class="grid-stat">{"".join(html_stat)}</div>', unsafe_allow_html=True)
st.write("")


# ----------------------------------------------------------------------------------------------
# 6. Kiri: kuesioner | Kanan: panel hasil prediksi
# ----------------------------------------------------------------------------------------------
kol_kiri, kol_kanan = st.columns([1.55, 1], gap="large")

with kol_kiri:
    for q, pertanyaan in VARK_PERTANYAAN.items():
        with st.container(border=True, key=f"kartu_q{q:02d}"):
            n_pilih = len(pilihan[q])
            selesai = n_pilih > 0
            chip = (f'<span class="chip chip-hijau">{n_pilih} opsi dipilih</span>' if selesai
                    else '<span class="chip chip-abu">Belum dijawab</span>')
            st.markdown(f'<div class="q-kepala"><div class="q-no{" selesai" if selesai else ""}">{q}</div>'
                        f'<div class="q-teks">{pertanyaan}</div><div class="q-chip">{chip}</div></div>',
                        unsafe_allow_html=True)
            for i, opsi in enumerate(VARK_OPSI[q]):
                st.checkbox(opsi, key=kunci(q, i), width="stretch")


def html_kosong():
    daftar = ", ".join(str(q) for q in belum_dijawab)
    return (f'<div class="kosong"><div class="kosong-ikon">{ikon("daftar", 24)}</div>'
            f'<h4>Jawab semua pertanyaan dulu</h4>'
            f'<p>Sudah {n_terjawab} dari {n_total} terjawab. Yang belum: nomor {daftar}.</p>'
            f'<div class="bar" style="max-width:260px;margin:1rem auto 0"><span style="width:{persen}%;background:var(--oranye)"></span></div>'
            '<p style="margin-top:.9rem;font-size:.8rem">Model dilatih dengan jawaban lengkap, jadi pertanyaan kosong '
            'tidak diisi otomatis oleh aplikasi.</p></div>')


with kol_kanan:
    with st.container(key="panel_hasil"):
        st.markdown('<div class="panel-judul">Hasil prediksi</div>'
                    '<div class="panel-sub">Dihitung oleh model tersimpan setiap kali jawabanmu berubah.</div>',
                    unsafe_allow_html=True)
        tab_klf, tab_reg = st.tabs(["Klasifikasi VARK", "Regresi IPK"])

        # ---------------- Tab Klasifikasi ----------------
        with tab_klf:
            if x_input is None:
                st.markdown(html_kosong(), unsafe_allow_html=True)
            else:
                kategori = str(model_klf.predict(x_input)[0])   # output dari model tersimpan
                warna = WARNA_KATEGORI[kategori]
                catatan_seri = ('<div class="catatan">Seri berarti jawabanmu tidak condong ke satu modalitas saja. '
                                'Banyak orang memang memakai lebih dari satu cara belajar.</div>'
                                if kategori == "Seri" else "")
                st.markdown(
                    f'<div class="hasil"><div class="hasil-chip"><span class="chip chip-navy">Framework {FRAMEWORK}</span></div>'
                    f'<div class="hasil-atas">'
                    f'<div class="avatar" style="background:{warna}">{INISIAL_KATEGORI[kategori]}</div>'
                    f'<div><div class="hasil-kecil">Kategori menurut model</div>'
                    f'<div class="hasil-besar" style="color:{warna}">{kategori}</div></div></div>'
                    f'<div class="hasil-teks">{PENJELASAN_KATEGORI[kategori]}</div>{catatan_seri}</div>',
                    unsafe_allow_html=True)

                jumlah = jumlah_pilihan_per_modalitas(x_input)
                baris = "".join(
                    f'<div class="profil-baris"><span>{nama}</span><div class="bar"><span style="width:{nilai / 16 * 100:.0f}%;'
                    f'background:{WARNA_KATEGORI[nama]}"></span></div><span class="profil-nilai">{nilai}/16</span></div>'
                    for nama, nilai in jumlah.items())
                urut = sorted(jumlah.values(), reverse=True)
                selisih = urut[0] - urut[1]
                keterangan = ("Selisihnya tipis, jadi hasil model lebih rawan keliru. Di data test, hampir semua "
                              "kesalahan model terjadi pada selisih 0-1." if selisih <= 1 else
                              "Selisihnya cukup jelas, jadi hasil model biasanya lebih bisa dipercaya.")
                m = card_klf["metrik_test"]
                st.markdown(
                    '<div class="sub-judul">Profil jawabanmu</div>' + baris +
                    f'<div class="catatan">Ini hitungan sederhana dari centangmu, <b>bukan output model</b>. '
                    f'Selisih modalitas terkuat dengan urutan kedua: <b>{selisih}</b>. {keterangan}</div>'
                    f'<div class="metrik-mini"><div><b>{m["accuracy test"]:.0%}</b><span>accuracy di data test</span></div>'
                    f'<div><b>{m["macro-F1 test"]:.2f}</b><span>macro-F1 di data test</span></div></div>',
                    unsafe_allow_html=True)

        # ---------------- Tab Regresi IPK ----------------
        with tab_reg:
            st.markdown(f'<div class="banner banner-kuning">{ikon("info", 17)}<span><b>Bukan IPK resmi</b> dan bukan '
                        'jaminan akademik. IPK dipengaruhi banyak hal yang tidak ditanyakan di kuesioner ini.</span></div>',
                        unsafe_allow_html=True)
            if x_input is None:
                st.markdown(html_kosong(), unsafe_allow_html=True)
            else:
                ipk = float(model_reg.predict(x_input)[0])   # output dari model tersimpan
                status_rentang = cek_rentang_ipk(ipk)
                di_luar = status_rentang.startswith("PERINGATAN")
                rata = card_reg["ipk_train"]["rata_rata"]
                pos_ipk = min(max(ipk / 4 * 100, 0), 100)
                pos_rata = rata / 4 * 100
                chip_status = ('<span class="chip" style="background:var(--merah-muda);color:var(--merah)">Di luar rentang 0-4</span>'
                               if di_luar else '<span class="chip chip-hijau">Dalam rentang 0-4</span>')
                st.markdown(
                    f'<div class="hasil"><div class="hasil-chip">{chip_status}</div>'
                    f'<div class="hasil-kecil">Perkiraan IPK menurut model</div>'
                    f'<div style="white-space:nowrap"><span class="ipk-angka">{ipk:.2f}</span> '
                    f'<span class="ipk-skala">/ 4.00</span></div>'
                    f'<div class="skala"><div class="skala-rata" style="left:{pos_rata:.1f}%"></div>'
                    f'<div class="skala-titik" style="left:{pos_ipk:.1f}%"></div>'
                    f'<div class="skala-label" style="left:{pos_ipk:.1f}%">{ipk:.2f}</div></div>'
                    '<div class="skala-angka"><span>0</span><span>1</span><span>2</span><span>3</span><span>4</span></div>'
                    '<div class="legenda"><span><i style="background:var(--oranye)"></i>Prediksi kamu</span>'
                    f'<span><i style="background:var(--navy);border-radius:2px;width:3px"></i>Rata-rata data training ({rata:.2f})</span></div>'
                    '</div>', unsafe_allow_html=True)
                if di_luar:
                    st.markdown(f'<div class="banner banner-merah" style="margin-top:.8rem">{ikon("seru", 17)}<span>'
                                f'<b>{status_rentang}.</b> Angka ini tidak masuk akal sebagai IPK dan sebaiknya diabaikan.'
                                '</span></div>', unsafe_allow_html=True)
                mt, mb = card_reg["metrik_test"], card_reg["metrik_test_baseline"]
                st.markdown(
                    '<div class="sub-judul">Cara membaca angka ini</div>'
                    f'<div class="hasil-teks" style="margin-top:0">Di data test, model ini rata-rata meleset '
                    f'<b>±{mt["MAE test"]:.2f} poin IPK</b>. Itu hampir sama dengan sekadar menebak rata-rata '
                    f'(±{mb["MAE test"]:.2f}). Artinya jawaban VARK hampir tidak memberi informasi tentang IPK, '
                    'sehingga prediksi akan selalu berada di sekitar rata-rata.</div>'
                    f'<div class="metrik-mini"><div><b>±{mt["MAE test"]:.2f}</b><span>MAE model</span></div>'
                    f'<div><b>±{mb["MAE test"]:.2f}</b><span>MAE tebak rata-rata</span></div></div>',
                    unsafe_allow_html=True)

        with st.expander("Tentang model yang dipakai"):
            pk, pr = card_klf["parameter"], card_reg["parameter"]
            st.markdown(
                '<table class="tabel-model">'
                f'<tr><td>Klasifikasi</td><td><b>{card_klf["nama_model"]}</b><br>kernel {pk["kernel"]}, C={pk["C"]}, '
                f'gamma={pk["gamma"]}, class_weight={pk["class_weight"]}, dengan StandardScaler</td></tr>'
                f'<tr><td>Regresi IPK</td><td><b>{card_reg["nama_model"]}</b><br>{pr["n_estimators"]} pohon, '
                f'min_samples_leaf={pr["min_samples_leaf"]}, max_features={pr["max_features"]}, max_depth={pr["max_depth"]}</td></tr>'
                f'<tr><td>Data</td><td>Train {card_klf["data"]["n_train"]} / test {card_klf["data"]["n_test"]} (klasifikasi), '
                f'train {card_reg["data"]["n_train"]} / test {card_reg["data"]["n_test"]} (regresi); split 80:20, random_state=42</td></tr>'
                '<tr><td>Input</td><td>64 indikator 0/1 (16 pertanyaan × 4 opsi), dikodekan dengan fungsi '
                '<code>encode_vark</code> yang sama dengan notebook</td></tr>'
                '<tr><td>Kategori Seri</td><td>Aturan operasional tugas (dua atau lebih modalitas sama kuat), '
                'bukan gaya belajar kelima</td></tr>'
                '</table>', unsafe_allow_html=True)
            st.caption("Kedua model dimuat dari file .joblib yang disimpan di notebook. Aplikasi tidak melatih ulang "
                       "dan tidak memakai aturan hard-coded.")
            versi_training = card_klf["versi"]["scikit-learn"]
            if sklearn.__version__ != versi_training:
                st.warning(f"Versi scikit-learn di server ({sklearn.__version__}) berbeda dengan saat training "
                           f"({versi_training}). Pasang versi sesuai requirements.txt agar hasil identik.")
