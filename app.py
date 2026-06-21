import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,f1_score, precision_score, recall_score,)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset_ispu_jakarta.csv"
MODEL_DIR = "saved_models"
FITUR_POLUTAN = ["pm25", "pm10", "so2", "co", "o3", "no2"]
VALID_LABELS = ["BAIK", "SEDANG", "TIDAK SEHAT"]
COLS_TO_DROP = ["tanggal", "periode_data", "stasiun", "parameter_pencemar_kritis", "max"]
MODEL_FILES = {
    "Random Forest": "best_random_forest_model.pkl",
    "KNN": "best_knn_model.pkl",
    "SVM": "best_svm_model.pkl",
}
PENJELASAN_POLUTAN = {
    "pm25": {
        "label": "PM2.5",
        "satuan": "µg/m³",
        "deskripsi": "Partikulat halus berukuran kurang dari atau sama dengan 2.5 mikrometer, berasal dari emisi kendaraan, asap pembakaran, dan industri. Karena ukurannya sangat kecil, partikel ini bisa masuk jauh ke saluran pernapasan dan paru-paru.",
        "min": 0.0, "max": 500.0, "default": 50.0, "step": 1.0,
        "alasan": "Batas max 500 mengikuti rentang indeks ISPU nasional (kategori berbahaya berada di atas 300), sedangkan 0 adalah nilai minimum yang mungkin.",
    },
    "pm10": {
        "label": "PM10",
        "satuan": "µg/m³",
        "deskripsi": "Partikulat kasar berukuran kurang dari atau sama dengan 10 mikrometer, contohnya debu jalanan, serbuk konstruksi, dan abu. Konsentrasinya biasanya lebih tinggi dari PM2.5 karena mencakup partikel yang lebih besar juga.",
        "min": 0.0, "max": 600.0, "default": 80.0, "step": 1.0,
        "alasan": "Mengikuti rentang baku mutu ISPU untuk PM10, dengan batas atas sedikit lebih tinggi dari PM2.5 karena secara historis nilai PM10 cenderung lebih besar.",
    },
    "so2": {
        "label": "SO2 (Sulfur Dioksida)",
        "satuan": "µg/m³",
        "deskripsi": "Gas hasil pembakaran bahan bakar fosil yang mengandung sulfur, seperti batu bara dan solar industri. Sumber utamanya adalah pembangkit listrik dan kendaraan diesel.",
        "min": 0.0, "max": 800.0, "default": 30.0, "step": 1.0,
        "alasan": "Rentang disesuaikan dengan ambang ISPU untuk SO2, di mana nilai di atas 400 umumnya sudah masuk kategori berbahaya.",
    },
    "co": {
        "label": "CO (Karbon Monoksida)",
        "satuan": "µg/m³",
        "deskripsi": "Gas tidak berwarna dan tidak berbau hasil pembakaran tidak sempurna, paling banyak berasal dari knalpot kendaraan bermotor di area padat lalu lintas.",
        "min": 0.0, "max": 30000.0, "default": 4000.0, "step": 100.0,
        "alasan": "Satuan CO dalam dataset jauh lebih besar dibanding polutan lain karena konsentrasinya memang biasa terukur dalam ribuan µg/m³, jadi batas atas dibuat jauh lebih tinggi mengikuti pola data asli.",
    },
    "o3": {
        "label": "O3 (Ozon Permukaan)",
        "satuan": "µg/m³",
        "deskripsi": "Ozon di lapisan udara dekat permukaan tanah, terbentuk dari reaksi kimia antara sinar matahari dengan polutan kendaraan dan industri. Berbeda dengan ozon di lapisan atmosfer atas yang melindungi bumi.",
        "min": 0.0, "max": 500.0, "default": 60.0, "step": 1.0,
        "alasan": "Mengikuti rentang ambang ISPU untuk ozon permukaan, dengan nilai tinggi biasanya muncul siang hari saat cuaca cerah dan terik.",
    },
    "no2": {
        "label": "NO2 (Nitrogen Dioksida)",
        "satuan": "µg/m³",
        "deskripsi": "Gas hasil pembakaran bahan bakar pada suhu tinggi, sumber utamanya adalah kendaraan bermotor dan pembangkit listrik. Sering jadi indikator polusi lalu lintas.",
        "min": 0.0, "max": 400.0, "default": 40.0, "step": 1.0,
        "alasan": "Batas atas mengikuti ambang ISPU untuk NO2, karena di atas itu jarang ditemukan pada data pemantauan harian.",
    },
}

PESAN_KATEGORI = {
    "BAIK": ("success", "BAIK",
            "Kualitas udara berada pada kondisi baik tidak memberi efek negatif terhadap manusia, hewan, atau tumbuhan, dan aman untuk aktivitas luar ruangan. Konsentrasi polutan masih berada pada tingkat yang relatif rendah."),
    "SEDANG": ("warning", "SEDANG",
            "Kualitas udara berada pada kategori sedang. Secara umum masih aman, namun kelompok sensitif seperti anak-anak, lansia, dan penderita gangguan pernapasan disarankan mengurangi aktivitas luar ruangan."),
    "TIDAK SEHAT": ("error", "TIDAK SEHAT",
            "Kualitas udara berada pada kategori tidak sehat. Tingginya konsentrasi beberapa polutan dapat meningkatkan risiko gangguan kesehatan sehingga disarankan mengurangi aktivitas luar ruangan dan menggunakan masker."),
}


# load model
@st.cache_resource
def muat_model():
    return {
        "models": {
            nama: joblib.load(os.path.join(MODEL_DIR, fname))
            for nama, fname in MODEL_FILES.items()
        },
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "le": joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl")),
    }

# load dan preprocess dataset
@st.cache_data
def load_and_preprocess_dataset():
    """Mengulang langkah pembersihan data persis seperti di notebook.
    Tidak ada training model di sini, cuma menyiapkan data."""
    df_raw = pd.read_csv(DATASET_PATH)
    df_clean = df_raw[df_raw["kategori"].isin(VALID_LABELS)].copy()
    df_clean = df_clean.drop(columns=COLS_TO_DROP, errors="ignore")
    df_clean = df_clean.sort_values("kategori").reset_index(drop=True)
    for col in FITUR_POLUTAN:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].interpolate(method="linear", limit_direction="both")
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    info = {
        "n_raw": len(df_raw),
        "n_clean": len(df_clean),
        "fitur_dipakai": [f for f in FITUR_POLUTAN if f in df_clean.columns],
    }
    return df_raw, df_clean, info


def buat_ulang_data_test(df_clean, le, fitur_dipakai):
    """Mengulang pembagian data latih/test persis seperti di notebook (pakai pengaturan
    yang sama), supaya model yang sudah tersimpan bisa dites lagi tanpa dilatih ulang."""
    y = le.transform(df_clean["kategori"])
    X = df_clean[fitur_dipakai].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test


def hitung_metrik(y_true, y_pred, nama_model):
    return {
        "Model": nama_model,
        "Akurasi (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Presisi (%)": round(precision_score(y_true, y_pred, average="weighted") * 100, 2),
        "Recall (%)": round(recall_score(y_true, y_pred, average="weighted") * 100, 2),
        "F1-Score (%)": round(f1_score(y_true, y_pred, average="weighted") * 100, 2),
    }

# bagian visualisasi
def plot_distribusi(df_raw):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vc = df_raw["kategori"].value_counts()
    sns.barplot(x=vc.index, y=vc.values, palette="viridis", ax=ax)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_title("Distribusi Kategori ISPU (Sebelum Dibersihkan)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Kategori"); ax.set_ylabel("Jumlah"); ax.tick_params(labelsize=7)
    plt.xticks(rotation=10); plt.tight_layout()
    return fig


def plot_histogram_polutan(df_raw):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    colors = sns.color_palette("Set2", 6)
    for i, (col, color) in enumerate(zip(FITUR_POLUTAN, colors)):
        ax = axes[i // 3][i % 3]
        ax.hist(df_raw[col].dropna(), bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"Distribusi {col.upper()}", fontsize=9)
        ax.set_xlabel("Nilai", fontsize=7); ax.set_ylabel("Frekuensi", fontsize=7)
        ax.tick_params(labelsize=6)
    plt.suptitle("Distribusi Nilai Polutan Udara", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def plot_korelasi(df_raw):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    corr = df_raw[FITUR_POLUTAN].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, linewidths=0.5, square=True, ax=ax, annot_kws={"size": 7})
    ax.set_title("Matriks Korelasi Antar Fitur Polutan", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7); plt.tight_layout()
    return fig


def plot_perbandingan_metrik(hasil_df):
    df_plot = hasil_df.reset_index()
    metrik_cols = ["Akurasi (%)", "Presisi (%)", "Recall (%)", "F1-Score (%)"]
    colors_bar = ["#9BD1FD", "#8BF58F", "#F5CD90", "#FC9FBE"]
    x, width = np.arange(len(df_plot)), 0.18
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (col, color) in enumerate(zip(metrik_cols, colors_bar)):
        bars = ax.bar(x + i * width, df_plot[col], width, label=col, color=color, alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
            ax.set_xticks(x + width * 1.5); ax.set_xticklabels(df_plot["Model"], fontsize=9)
            ax.set_ylim(max(0, hasil_df.min().min() - 10), 108)
            ax.set_ylabel("Nilai Metrik (%)", fontsize=8)
            ax.set_title("Perbandingan Metrik Semua Model", fontsize=10, fontweight="bold")
            ax.legend(loc="lower right", fontsize=7); ax.yaxis.grid(True, alpha=0.4)
            ax.tick_params(labelsize=7); plt.tight_layout()
    return fig


def plot_confusion_matrix(preds, y_test, le):
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.5))
    axes = [axes] if len(preds) == 1 else axes
    for ax, (nama, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(nama, fontsize=10, pad=4)
        ax.tick_params(axis="x", rotation=15, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("Confusion Matrix Semua Model", fontsize=11, y=1.02, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(rf_model, fitur_dipakai):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_feat = [fitur_dipakai[i] for i in idx]
    sorted_imp = importances[idx]
    fig, ax = plt.subplots(figsize=(6, max(3, len(sorted_feat))))
    bars = ax.barh(sorted_feat, sorted_imp, color=sns.color_palette("viridis", len(sorted_feat)), edgecolor="white", height=0.6)
    for bar, val in zip(bars, sorted_imp):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.set_title("Feature Importance - Random Forest", fontsize=10, fontweight="bold")
        ax.invert_yaxis(); ax.set_xlim(0, max(sorted_imp) * 1.3); plt.tight_layout()
    return fig


def render(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# halaman hasil penelitian
def halaman_hasil_penelitian():
    st.title("Hasil Penelitian")
    st.markdown(
        "**Klasifikasi Kualitas Udara DKI Jakarta** menggunakan tiga algoritma: KNN, SVM, dan Random Forest "
        "dengan data sensor polutan dari stasiun pemantauan kualitas udara di Jakarta. "
    )

    model_data = muat_model()
    if not os.path.exists(DATASET_PATH):
        st.error(f"File `{DATASET_PATH}` tidak ditemukan di folder")
        return

    df_raw, df_clean, info = load_and_preprocess_dataset()
    le, scaler, models = model_data["le"], model_data["scaler"], model_data["models"]

    st.subheader("Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris Awal", info["n_raw"])
    c2.metric("Baris Setelah Dibersihkan", info["n_clean"])
    c3.metric("Baris Dihapus", info["n_raw"] - info["n_clean"])
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Eksplorasi Data")
    tab1, tab2, tab3 = st.tabs(["Distribusi Kategori", "Distribusi Polutan", "Korelasi Fitur"])
    with tab1:
        col, _ = st.columns([1, 1]); col.pyplot(plot_distribusi(df_raw), use_container_width=False)
    with tab2:
        render(plot_histogram_polutan(df_raw))
    with tab3:
        col, _ = st.columns([1, 1]); col.pyplot(plot_korelasi(df_raw), use_container_width=False)

    st.subheader("Pembersihan Data")
    st.write(
        f"- Filter hanya kategori valid `{', '.join(VALID_LABELS)}`\n"
        f"- Hapus kolom yang tidak dipakai sebagai fitur: `{', '.join(COLS_TO_DROP)}`\n"
        f"- Isi nilai kosong pada fitur polutan dengan interpolasi linear\n"
        f"- Hapus baris duplikat"
    )
    st.markdown("**Mapping Label**")
    st.dataframe(pd.DataFrame({"Label Asli": le.classes_, "Kode Numerik": le.transform(le.classes_)}),
                 use_container_width=True, hide_index=True)

    st.subheader("Hasil Evaluasi Model (dari Notebook)")
    X_test, y_test = buat_ulang_data_test(df_clean, le, info["fitur_dipakai"])
    X_test_scaled = scaler.transform(X_test)

    preds = {nama: model.predict(X_test_scaled) for nama, model in models.items()}
    hasil_df = pd.DataFrame(
        [hitung_metrik(y_test, y_pred, nama) for nama, y_pred in preds.items()]
    ).set_index("Model")
    st.dataframe(hasil_df.style.highlight_max(axis=0, color="#00ff33"), use_container_width=True)

    best_name = hasil_df["F1-Score (%)"].idxmax()
    st.success(f"Model terbaik berdasarkan F1-Score: **{best_name}** ({hasil_df.loc[best_name, 'F1-Score (%)']}%)")

    st.markdown("**Perbandingan Metrik**")
    col, _ = st.columns([3, 1]); col.pyplot(plot_perbandingan_metrik(hasil_df), use_container_width=False)

    st.markdown("**Confusion Matrix**")
    col, _ = st.columns([3, 1]); col.pyplot(plot_confusion_matrix(preds, y_test, le), use_container_width=False)

    if "Random Forest" in models:
        st.markdown("**Feature Importance (Random Forest)**")
        col, _ = st.columns([2, 1])
        col.pyplot(plot_feature_importance(models["Random Forest"], info["fitur_dipakai"]),
                   use_container_width=False)


# halaman prediksi
def halaman_prediksi():
    st.title("Prediksi Kualitas Udara")
    st.write(
        "Pilih stasiun pemantauan dan tanggal untuk melihat hasil prediksi kualitas "
        "udara berdasarkan data sensor polutan "
    )

    model_data = muat_model()
    if not os.path.exists(DATASET_PATH):
        st.error(f"File `{DATASET_PATH}` tidak ditemukan di folder aplikasi.")
        return

    models, scaler, le = model_data["models"], model_data["scaler"], model_data["le"]

    nama_model = st.selectbox(
        "Pilih Model", list(models.keys()),
        index=list(models.keys()).index("Random Forest") if "Random Forest" in models else 0,
    )
    model = models[nama_model]

    mode_input = st.radio(
        "Sumber Data", ["Pilih dari Dataset", "Input Manual"],
        horizontal=True,
        help="Pilih dari Dataset memakai data sensor asli pada tanggal & stasiun tertentu. "
             "Input Manual memungkinkan memasukkan nilai polutan sendiri secara bebas, "
             "misalnya untuk simulasi atau pengujian.",
    )

    data_baru = None

    if mode_input == "Pilih dari Dataset":
        df = pd.read_csv(DATASET_PATH).dropna(subset=FITUR_POLUTAN)
        df["tanggal_lengkap"] = pd.to_datetime(
            df["periode_data"].astype(str) + df["tanggal"].astype(str).str.zfill(2),
            format="%Y%m%d", errors="coerce",
        )
        df = df.dropna(subset=["tanggal_lengkap"])

        stasiun = st.selectbox("Pilih Stasiun", sorted(df["stasiun"].unique()))
        df_stasiun = df[df["stasiun"] == stasiun].sort_values("tanggal_lengkap")
        tanggal = st.selectbox(
            "Pilih Tanggal", df_stasiun["tanggal_lengkap"].dt.date.tolist(),
            format_func=lambda d: d.strftime("%d %B %Y"),
        )
        data_pilih = df_stasiun[df_stasiun["tanggal_lengkap"].dt.date == tanggal].iloc[0]

        st.subheader("Data Sensor Polutan")
        cols = st.columns(3)
        pasangan = [("PM25", "pm25", "PM10", "pm10"), ("SO2", "so2", "CO", "co"), ("O3", "o3", "NO2", "no2")]
        for col, (label_a, key_a, label_b, key_b) in zip(cols, pasangan):
            col.metric(label_a, data_pilih[key_a])
            col.metric(label_b, data_pilih[key_b])

        data_baru = pd.DataFrame([{col: data_pilih[col] for col in FITUR_POLUTAN}])

    else:
        st.subheader("Input Manual Parameter Polutan")
        st.caption(
            "Masukkan nilai untuk setiap parameter polutan secara bebas sesuai kebutuhan "
            "pengujian. Penjelasan dan rentang masing-masing parameter ada di bawah setiap kolom."
        )

        with st.expander("Penjelasan Parameter & Alasan Rentang Nilai", expanded=False):
            for fitur in FITUR_POLUTAN:
                info_p = PENJELASAN_POLUTAN[fitur]
                st.markdown(f"**{info_p['label']}** ({info_p['satuan']})")
                st.write(info_p["deskripsi"])
                st.caption(
                    f"Rentang input: {info_p['min']:.0f} – {info_p['max']:.0f} {info_p['satuan']}. "
                    f"Alasan: {info_p['alasan']}"
                )
                st.markdown("---")

        nilai_manual = {}
        cols_manual = st.columns(3)
        for i, fitur in enumerate(FITUR_POLUTAN):
            info_p = PENJELASAN_POLUTAN[fitur]
            col = cols_manual[i % 3]
            nilai_manual[fitur] = col.number_input(
                f"{info_p['label']} ({info_p['satuan']})",
                min_value=info_p["min"],
                max_value=info_p["max"],
                value=info_p["default"],
                step=info_p["step"],
                help=f"{info_p['deskripsi']} Rentang: {info_p['min']:.0f}–{info_p['max']:.0f} "
                     f"{info_p['satuan']}. {info_p['alasan']}",
                key=f"manual_{fitur}",
            )

        data_baru = pd.DataFrame([nilai_manual])[FITUR_POLUTAN]

    if st.button("Prediksi", type="primary"):
        hasil = model.predict(scaler.transform(data_baru))
        kategori = le.inverse_transform(hasil)[0]

        st.subheader("Hasil Prediksi")
        st.caption(f"Model yang digunakan: **{nama_model}**")
        level, judul, deskripsi = PESAN_KATEGORI.get(kategori, PESAN_KATEGORI["TIDAK SEHAT"])
        getattr(st, level)(judul)
        st.write(deskripsi)

        st.divider()
        st.subheader("Analisis Data Polutan")
        polutan_tertinggi = data_baru.iloc[0].idxmax().upper()
        nilai_tertinggi = data_baru.iloc[0].max()
        st.info(
            f"Polutan dominan pada data ini adalah {polutan_tertinggi} dengan nilai "
            f"{nilai_tertinggi:.0f}. Parameter tersebut menjadi salah satu faktor yang "
            f"memengaruhi hasil klasifikasi kualitas udara."
        )


# buat konfigurasi halaman dan navigasi
st.set_page_config(page_title="Klasifikasi Kualitas Udara DKI Jakarta", layout="wide")

with st.sidebar:
    st.title("Klasifikasi Kualitas Udara")
    halaman = st.radio("Navigasi", ["Hasil Penelitian", "Prediksi"], label_visibility="collapsed")
    st.divider()
    st.caption("Model KNN, SVM, dan Random Forest sudah dilatih dan dioptimalkan di notebook, lalu dimuat langsung di sini.")

if halaman == "Hasil Penelitian":
    halaman_hasil_penelitian()
else:
    halaman_prediksi()