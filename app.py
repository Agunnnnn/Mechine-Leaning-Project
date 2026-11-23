from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Klasifikasi Nilai Akademik Siswa",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "student-gabungan.csv"
    if not data_path.exists():
        data_path = "student-gabungan.csv"
    df = pd.read_csv(data_path)
    
    def categorize_g3(score):
        if score >= 16:
            return "A (Sangat Baik)"
        elif score >= 14:
            return "B (Baik)"
        elif score >= 12:
            return "C (Cukup)"
        elif score >= 10:
            return "D (Kurang)"
        else:
            return "E (Gagal)"

    if "predikat" not in df.columns:
        df["predikat"] = df["G3"].apply(categorize_g3)
        
    return df

try:
    df = load_data()
    datasets = {"Student Performance": df}
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    df = pd.DataFrame()
    datasets = {}

# --------------------------------------------------
# Sidebar Navigation & Upload
# --------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Tambahkan dataset CSV",
    type=["csv"],
    accept_multiple_files=True,
    key="dataset_uploader"
)

for file_index, uploaded_file in enumerate(uploaded_files, start=1):
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        if dataset_name in datasets:
            dataset_name = f"{dataset_name} ({file_index})"
        datasets[dataset_name] = uploaded_df
    except Exception as error:
        st.sidebar.error(f"Gagal membaca {uploaded_file.name}: {error}")

selected_dataset = st.sidebar.selectbox("Dataset aktif", list(datasets.keys()))
if selected_dataset:
    df_active = datasets[selected_dataset]
else:
    df_active = df

st.sidebar.title("📊 Analisis & Pemodelan")
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Beranda", "Dataset", "Statistik", "Visualisasi", "Grafik", "Pemodelan Machine Learning", "🔮 Prediksi Siswa Baru"]
)

# --------------------------------------------------
# Halaman Beranda
# --------------------------------------------------
if menu == "Beranda":
    st.title(f"🎓 Dashboard: {selected_dataset}")
    st.header("Analisis & Klasifikasi Predikat Nilai Akademik Siswa")
    st.write("Gunakan menu di samping untuk melihat data, statistik, visualisasi, dan hasil pemodelan Machine Learning.")
    
    st.subheader("Informasi Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", f"{len(df_active):,}")
    col2.metric("Jumlah Fitur", len(df_active.columns))
    col3.metric("Dataset Aktif", selected_dataset)

# --------------------------------------------------
# Halaman Dataset
# --------------------------------------------------
elif menu == "Dataset":
    st.title(f"📁 Dataset: {selected_dataset}")
    st.write("Berikut isi dataset yang sedang dipilih.")
    st.dataframe(df_active, use_container_width=True)

# --------------------------------------------------
# Halaman Statistik
# --------------------------------------------------
elif menu == "Statistik":
    st.title(f"📈 Statistik: {selected_dataset}")
    st.write("Statistik deskriptif untuk kolom numerik dan kategorikal.")
    st.dataframe(df_active.describe(include="all").transpose(), use_container_width=True)

# --------------------------------------------------
# Halaman Visualisasi
# --------------------------------------------------
elif menu == "Visualisasi":
    st.title(f"🎨 Visualisasi: {selected_dataset}")
    
    if "predikat" in df_active.columns:
        st.subheader("Distribusi Predikat Nilai Akademik (Target)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df_active, x="predikat", order=["A (Sangat Baik)", "B (Baik)", "C (Cukup)", "D (Kurang)", "E (Gagal)"], palette="viridis", ax=ax)
        ax.set_title("Distribusi Kelas Predikat")
        ax.set_xlabel("Predikat")
        ax.set_ylabel("Jumlah Siswa")
        st.pyplot(fig)

    numeric_columns = df_active.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        st.subheader("Distribusi Kolom Numerik")
        histogram_column = st.selectbox("Pilih kolom numerik", numeric_columns)
        fig, ax = plt.subplots(figsize=(8, 4))
        df_active[histogram_column].dropna().plot(kind="hist", bins=20, ax=ax, color="skyblue", edgecolor="black")
        ax.set_title(f"Distribusi {histogram_column}")
        ax.set_xlabel(histogram_column)
        st.pyplot(fig)

    if len(numeric_columns) > 1:
        st.subheader("Hubungan Antar Kolom Numerik")
        x_col = st.selectbox("Sumbu X", numeric_columns, index=0)
        y_col = st.selectbox("Sumbu Y", numeric_columns, index=1 if numeric_columns[0] == x_col else 0)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df_active[x_col], df_active[y_col], alpha=0.6, color="purple")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig)

# --------------------------------------------------
# Halaman Grafik (Menu Baru)
# --------------------------------------------------
elif menu == "Grafik":
    st.title(f"📊 Berbagai Macam Grafik: {selected_dataset}")
    st.write("Eksplorasi data dengan beragam tipe visualisasi (Pie Chart, Bar Chart, Boxplot, dan Correlation Heatmap).")
    
    cat_columns = df_active.select_dtypes(include=["object", "category"]).columns.tolist()
    num_columns = df_active.select_dtypes(include="number").columns.tolist()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🥧 Pie Chart & Donut", "📊 Bar Chart", "📦 Boxplot & Outlier", "🔥 Correlation Heatmap"])
    
    with tab1:
        st.subheader("Pie Chart & Donut Chart (Komposisi Kategori)")
        if cat_columns:
            selected_cat = st.selectbox("Pilih Kolom Kategorikal", cat_columns, key="pie_cat_col")
            chart_style = st.radio("Tipe Chart", ["Pie Chart", "Donut Chart"], horizontal=True, key="pie_style")
            
            val_counts = df_active[selected_cat].value_counts().dropna()
            
            fig, ax = plt.subplots(figsize=(7, 7))
            colors = sns.color_palette("pastel")[0:len(val_counts)]
            
            wedges, texts, autotexts = ax.pie(
                val_counts.values,
                labels=val_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                pctdistance=0.8 if chart_style == "Donut Chart" else 0.6
            )
            
            if chart_style == "Donut Chart":
                centre_circle = plt.Circle((0, 0), 0.55, fc="white")
                fig.gca().add_artist(centre_circle)
                
            plt.setp(autotexts, size=10, weight="bold")
            ax.set_title(f"Proporsi Berdasarkan {selected_cat}", fontsize=14, pad=20)
            st.pyplot(fig)
            
            with st.expander(f"Lihat Tabel Angka: {selected_cat}"):
                summary_pie = pd.DataFrame({
                    "Jumlah Siswa": val_counts,
                    "Persentase (%)": (val_counts / val_counts.sum() * 100).round(2)
                })
                st.dataframe(summary_pie)
        else:
            st.info("Tidak ditemukan kolom kategorikal pada dataset ini.")

    with tab2:
        st.subheader("Bar Chart Frekuensi Kategori")
        if cat_columns:
            bar_col = st.selectbox("Pilih Kolom Kategorikal", cat_columns, key="bar_cat_col")
            orientation = st.radio("Orientasi Bar", ["Vertikal", "Horizontal"], horizontal=True, key="bar_orient")
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            val_counts = df_active[bar_col].value_counts().dropna()
            
            if orientation == "Vertikal":
                sns.barplot(x=val_counts.index, y=val_counts.values, palette="mako", ax=ax)
                ax.set_xlabel(bar_col)
                ax.set_ylabel("Jumlah")
                plt.xticks(rotation=45 if len(val_counts) > 5 else 0)
            else:
                sns.barplot(x=val_counts.values, y=val_counts.index, palette="mako", ax=ax)
                ax.set_xlabel("Jumlah")
                ax.set_ylabel(bar_col)
                
            ax.set_title(f"Distribusi Frekuensi {bar_col}")
            st.pyplot(fig)
        else:
            st.info("Tidak ditemukan kolom kategorikal pada dataset ini.")

    with tab3:
        st.subheader("Boxplot & Deteksi Outlier")
        if num_columns:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                selected_num = st.selectbox("Pilih Kolom Numerik", num_columns, key="box_num_col")
            with col_b2:
                group_options = ["None"] + cat_columns
                selected_group = st.selectbox("Kelompokkan Berdasarkan (Opsional)", group_options, key="box_group_col")
                
            fig, ax = plt.subplots(figsize=(8, 5))
            if selected_group == "None":
                sns.boxplot(y=df_active[selected_num], color="skyblue", ax=ax)
                ax.set_title(f"Boxplot {selected_num}")
            else:
                sns.boxplot(x=df_active[selected_group], y=df_active[selected_num], palette="Set2", ax=ax)
                ax.set_title(f"Boxplot {selected_num} berdasarkan {selected_group}")
                plt.xticks(rotation=45 if df_active[selected_group].nunique() > 5 else 0)
                
            st.pyplot(fig)
        else:
            st.info("Tidak ditemukan kolom numerik pada dataset ini.")

    with tab4:
        st.subheader("Correlation Heatmap (Matriks Korelasi)")
        if len(num_columns) > 1:
            corr_matrix = df_active[num_columns].corr()
            fig_size = max(8, len(num_columns) * 0.6)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.75))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5)
            ax.set_title("Matriks Korelasi Antar Fitur Numerik", fontsize=14, pad=15)
            st.pyplot(fig)
        else:
            st.info("Dibutuhkan minimal 2 kolom numerik untuk menampilkan correlation heatmap.")

# --------------------------------------------------
# Halaman Pemodelan Machine Learning
# --------------------------------------------------
elif menu == "Pemodelan Machine Learning":
    st.title("🤖 Klasifikasi Random Forest dengan PCA")
    st.write("Pengujian klasifikasi predikat nilai akademik dengan mengeliminasi data leakage (G1, G2, G3) dan mereduksi dimensi dengan PCA.")
    
    if "predikat" not in df_active.columns:
        st.warning("Dataset tidak memiliki kolom predikat untuk dijadikan target klasifikasi.")
    else:
        # Preprocessing Data
        X = df_active.drop(columns=[col for col in ["G1", "G2", "G3", "predikat"] if col in df_active.columns])
        y = df_active["predikat"]
        
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
            ]
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        col1, col2 = st.columns(2)
        n_components = col1.slider("Jumlah Komponen PCA", min_value=2, max_value=min(20, X_processed.shape[1]), value=10)
        n_estimators = col2.slider("Jumlah Trees (n_estimators)", min_value=10, max_value=200, value=100, step=10)
        
        if st.button("Jalankan Pemodelan"):
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_processed)
            
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
            
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf.fit(X_train, y_train)
            
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            st.success(f"Model Berhasil Dilatih! Akurasi Data Testing: **{acc * 100:.2f}%**")
            st.write(f"Total variance ratio yang dijelaskan oleh {n_components} komponen PCA: **{sum(pca.explained_variance_ratio_) * 100:.2f}%**")
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = sorted(y.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)

# --------------------------------------------------
# Halaman Prediksi Siswa Baru
# --------------------------------------------------
elif menu == "🔮 Prediksi Siswa Baru":
    st.title("🔮 Prediksi Predikat Nilai Akademik Siswa Baru")
    st.write("Masukkan data siswa di bawah ini, lalu klik **Prediksi** untuk mendapatkan predikat nilai akademik.")

    if "predikat" not in df_active.columns:
        st.warning("Dataset aktif tidak memiliki kolom 'predikat'. Gunakan dataset Student Performance.")
    else:
        # ----- Form Input -----
        st.subheader("📝 Data Siswa")

        col1, col2, col3 = st.columns(3)

        with col1:
            school    = st.selectbox("Sekolah (school)", ["GP", "MS"])
            sex       = st.selectbox("Jenis Kelamin (sex)", ["F", "M"])
            age       = st.slider("Usia (age)", 15, 22, 17)
            address   = st.selectbox("Alamat (address)", ["U", "R"], help="U=Urban, R=Rural")
            famsize   = st.selectbox("Ukuran Keluarga (famsize)", ["GT3", "LE3"], help="GT3 = >3 orang, LE3 = ≤3 orang")
            Pstatus   = st.selectbox("Status Orang Tua (Pstatus)", ["T", "A"], help="T=Tinggal bersama, A=Terpisah")
            Medu      = st.slider("Pendidikan Ibu (Medu) 0-4", 0, 4, 2)
            Fedu      = st.slider("Pendidikan Ayah (Fedu) 0-4", 0, 4, 2)
            Mjob      = st.selectbox("Pekerjaan Ibu (Mjob)", ["teacher", "health", "services", "at_home", "other"])
            Fjob      = st.selectbox("Pekerjaan Ayah (Fjob)", ["teacher", "health", "services", "at_home", "other"])

        with col2:
            reason    = st.selectbox("Alasan Memilih Sekolah (reason)", ["home", "reputation", "course", "other"])
            guardian  = st.selectbox("Wali (guardian)", ["mother", "father", "other"])
            traveltime= st.slider("Waktu Perjalanan ke Sekolah (traveltime) 1-4", 1, 4, 1, help="1=<15 menit, 4=>1 jam")
            studytime = st.slider("Waktu Belajar per Minggu (studytime) 1-4", 1, 4, 2, help="1=<2 jam, 4=>10 jam")
            failures  = st.slider("Jumlah Kegagalan Kelas (failures)", 0, 4, 0)
            schoolsup = st.selectbox("Dukungan Belajar Sekolah (schoolsup)", ["yes", "no"])
            famsup    = st.selectbox("Dukungan Belajar Keluarga (famsup)", ["yes", "no"])
            paid      = st.selectbox("Les Berbayar (paid)", ["yes", "no"])
            activities= st.selectbox("Kegiatan Ekstrakurikuler (activities)", ["yes", "no"])
            nursery   = st.selectbox("Pernah TK (nursery)", ["yes", "no"])

        with col3:
            higher    = st.selectbox("Ingin Kuliah (higher)", ["yes", "no"])
            internet  = st.selectbox("Punya Internet di Rumah (internet)", ["yes", "no"])
            romantic  = st.selectbox("Punya Pacar (romantic)", ["yes", "no"])
            famrel    = st.slider("Kualitas Hubungan Keluarga (famrel) 1-5", 1, 5, 4)
            freetime  = st.slider("Waktu Luang Setelah Sekolah (freetime) 1-5", 1, 5, 3)
            goout     = st.slider("Frekuensi Keluar Bersama Teman (goout) 1-5", 1, 5, 3)
            Dalc      = st.slider("Konsumsi Alkohol Hari Kerja (Dalc) 1-5", 1, 5, 1)
            Walc      = st.slider("Konsumsi Alkohol Akhir Pekan (Walc) 1-5", 1, 5, 1)
            health    = st.slider("Status Kesehatan (health) 1-5", 1, 5, 5)
            absences  = st.slider("Jumlah Absensi (absences)", 0, 93, 0)
            mapel     = st.selectbox("Mata Pelajaran (mapel)", sorted(df_active["mapel"].unique().tolist()) if "mapel" in df_active.columns else ["math", "por"])

        st.divider()

        if st.button("🔮 Prediksi Sekarang", use_container_width=True):
            # Susun input sebagai DataFrame dengan kolom yang sama persis dengan X training
            input_data = pd.DataFrame([{
                "school": school, "sex": sex, "age": age, "address": address,
                "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
                "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
                "traveltime": traveltime, "studytime": studytime, "failures": failures,
                "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
                "activities": activities, "nursery": nursery, "higher": higher,
                "internet": internet, "romantic": romantic, "famrel": famrel,
                "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
                "health": health, "absences": absences, "mapel": mapel
            }])

            # Latih ulang model menggunakan seluruh dataset aktif
            X_train_full = df_active.drop(columns=[c for c in ["G1", "G2", "G3", "predikat"] if c in df_active.columns])
            y_train_full = df_active["predikat"]

            num_cols_pred = X_train_full.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols_pred = X_train_full.select_dtypes(include=["object", "category"]).columns.tolist()

            preprocessor_pred = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols_pred),
                    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols_pred)
                ]
            )

            X_processed_full = preprocessor_pred.fit_transform(X_train_full)

            n_comp = min(10, X_processed_full.shape[1])
            pca_pred = PCA(n_components=n_comp)
            X_pca_full = pca_pred.fit_transform(X_processed_full)

            rf_pred = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_pred.fit(X_pca_full, y_train_full)

            # Transform data input
            # Pastikan kolom input sama dengan training
            for col in X_train_full.columns:
                if col not in input_data.columns:
                    input_data[col] = X_train_full[col].mode()[0]
            input_data = input_data[X_train_full.columns]

            X_input_processed = preprocessor_pred.transform(input_data)
            X_input_pca = pca_pred.transform(X_input_processed)

            hasil = rf_pred.predict(X_input_pca)[0]
            proba = rf_pred.predict_proba(X_input_pca)[0]
            kelas = rf_pred.classes_

            # Tampilkan hasil
            warna_map = {
                "A (Sangat Baik)": "🟢",
                "B (Baik)": "🔵",
                "C (Cukup)": "🟡",
                "D (Kurang)": "🟠",
                "E (Gagal)": "🔴"
            }
            ikon = warna_map.get(hasil, "⚪")

            st.success(f"### {ikon} Predikat Prediksi: **{hasil}**")

            st.subheader("📊 Probabilitas per Predikat")
            proba_df = pd.DataFrame({
                "Predikat": kelas,
                "Probabilitas (%)": (proba * 100).round(2)
            }).sort_values("Probabilitas (%)", ascending=False)

            fig_p, ax_p = plt.subplots(figsize=(8, 3))
            sns.barplot(data=proba_df, x="Predikat", y="Probabilitas (%)", palette="viridis", ax=ax_p)
            ax_p.set_title("Probabilitas Prediksi per Predikat")
            ax_p.set_ylabel("Probabilitas (%)")
            ax_p.bar_label(ax_p.containers[0], fmt="%.1f%%", padding=3)
            st.pyplot(fig_p)
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
