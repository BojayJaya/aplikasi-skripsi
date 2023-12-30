from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import  re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.write("""
<center><h3 style = "text-align: justify;">ANALISIS SENTIMEN JAMU MADURA MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE DAN QUERY EXPANSION RANKING</h3></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write("""<h3 style = "text-align: center;"><img src="https://asset.kompas.com/crops/78bBP1gjXGFghLuRKY-TrLpD7UI=/0x0:1000x667/750x500/data/photo/2020/09/19/5f660d3e0141f.jpg" width="120" height="120"></h3>""",unsafe_allow_html=True), 
            ["Home", "Dosen Pembimbing", "Dosen Penguji", "Dataset", "Akurasi", "Implementation", "Tentang Kami"],
            icons=['house', 'person','person', 'bar-chart', 'check-circle', 'check2-square', 'info-circle'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://cf.shopee.co.id/file/224536e9ed4a0e07d2981cc0789350ea" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Dosen Pembimbing":
        st.write("""<h4 style = "text-align: center;">Dosen Pembimbing</h4>""", unsafe_allow_html=True)
        
        # Membuat dua kolom, satu untuk Dosen Pembimbing I dan satu lagi untuk Dosen Pembimbing II
        col1, col2 = st.columns(2)

        # Menampilkan gambar Dosen Pembimbing I di kolom pertama
        col1.image('https://raw.githubusercontent.com/BojayJaya/aplikasi-skripsi/c63bd64edf281b146e25034a49afff81a99ba927/ibuk%20rika.png', width=180, caption='Dosen Pembimbing I')
        col1.write("#### <span style='font-size: smaller;'>Dr. Rika Yunitarini, ST., MT.</span>", unsafe_allow_html=True)

        # Menampilkan gambar Dosen Pembimbing II di kolom kedua
        col2.image('https://raw.githubusercontent.com/BojayJaya/aplikasi-skripsi/c63bd64edf281b146e25034a49afff81a99ba927/ibuk%20fifin.png', width=165, caption='Dosen Pembimbing II')
        col2.write("#### <span style='font-size: smaller;'>Fifin Ayu Mufarroha, S.Kom., M.Kom.</span>", unsafe_allow_html=True)

    elif selected == "Dosen Penguji":
        st.write("""<h4 style = "text-align: center;">Dosen Penguji</h4>""", unsafe_allow_html=True)
        
        # Membuat tiga kolom, untuk Dosen Penguji I, Dosen Penguji II, dan Dosen Penguji III
        col1, col2, col3 = st.columns(3)

        # Menampilkan gambar Dosen Penguji I di kolom pertama
        col1.image('https://raw.githubusercontent.com/BojayJaya/aplikasi-skripsi/143c91b72abdc33c1a86d1df68f305f35e417f69/bapak%20kautsar.png', width=170, caption='Dosen Penguji I')
        col1.write("#### <span style='font-size: smaller;'>Moch. Kautsar Sophan, S.Kom., M.MT.</span>", unsafe_allow_html=True)

        # Menampilkan gambar Dosen Penguji II di kolom kedua
        col2.image('https://raw.githubusercontent.com/BojayJaya/aplikasi-skripsi/143c91b72abdc33c1a86d1df68f305f35e417f69/ibuk%20rosida.png', width=160, caption='Dosen Penguji II')
        col2.write("#### <span style='font-size: smaller;'>Rosida Vivin Nahari, S.Kom., M.T.</span>", unsafe_allow_html=True)
        
        # Menampilkan gambar Dosen Penguji III di kolom ketiga
        col3.image('https://raw.githubusercontent.com/BojayJaya/aplikasi-skripsi/143c91b72abdc33c1a86d1df68f305f35e417f69/bapak%20dwi.png', width=160, caption='Dosen Penguji III')
        col3.write("#### <span style='font-size: smaller;'>Dwi Kuswanto, S.Pd.,M.T.</span>", unsafe_allow_html=True)

    elif selected == "Dataset":
        st.write("#### Deskripsi Dataset :")
        st.write(""" <p style = "text-align: justify;">Dataset ini berisi ulasan masyarakat terhadap jamu Madura. Selanjutnya, data ulasan ini akan diklasifikasikan ke dalam dua kategori sentimen yaitu positif dan negatif, kemudian dilakukan penerapan algoritma Support Vector Machine dan Seleksi Fitur Query Expansion Ranking untuk mengetahui nilai akurasinya.</p>""", unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset :")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses untuk mengubah teks yang tidak teratur menjadi lebih terstruktur, yang nantinya akan membantu dalam pengolahan data.</p>""", unsafe_allow_html=True)
        st.write("#### Tahapan Preprocessing Dataset :")
        st.write(""" 
        Tahapan preprocessing data melibatkan lima langkah sebagai berikut:
        1. **Case Folding**: Mengubah semua huruf menjadi huruf kecil.
        2. **Punctuation Removal**: Menghapus tanda baca dari teks.
        3. **Tokenizing**: Membagi teks menjadi kata-kata.
        4. **Stopword Removal**: Menghapus kata-kata umum yang tidak memberikan informasi penting.
        5. **Stemming**: Mengubah kata-kata menjadi bentuk dasarnya.

        Di bawah ini adalah contoh dari dataset sebelum dan setelah preprocessing:
        """, unsafe_allow_html=True)

        st.write("#### Dataset Sebelum Preprocessing :")
        dt_sblm_p = pd.read_csv("dt_sblm_p.csv")
        st.write(dt_sblm_p)

        st.write("##### Dataset Setelah Preprocessing :")
        dt_stlh_p = pd.read_csv("dt_stlh_p.csv")
        st.write(dt_stlh_p)

    elif selected == "Akurasi":

        # Menyusun data ke dalam DataFrame
        data = {'Pembagian Dataset': ['90:10', '80:20', '70:30', '60:40'],
                'Akurasi': [87, 91, 93, 89]}

        df_akurasi = pd.DataFrame(data)
        # Mengubah nilai akurasi ke dalam format persen
        # df_akurasi['Akurasi'] = df_akurasi['Akurasi'].apply(lambda x: f'{x*100:.2f}%')

        # Menampilkan judul grafik
        st.write("""<h6 style="text-align: center;">Grafik Akurasi Model SVM tanpa QER</h6>""", unsafe_allow_html=True)

        st.bar_chart(df_akurasi.set_index('Pembagian Dataset'), height=300)

        # Menyusun data ke dalam DataFrame
        data_25 = {'Pembagian Dataset': ['90:10', '80:20', '70:30', '60:40'],
                'Akurasi': [76, 84, 85, 83]}
        df_akurasi_25 = pd.DataFrame(data_25)

        data_50 = {'Pembagian Dataset': ['90:10', '80:20', '70:30', '60:40'],
                'Akurasi': [90, 92, 94, 87]}
        df_akurasi_50 = pd.DataFrame(data_50)

        data_75 = {'Pembagian Dataset': ['90:10', '80:20', '70:30', '60:40'],
                'Akurasi': [93, 92, 90, 89]}
        df_akurasi_75 = pd.DataFrame(data_75)

        data_100 = {'Pembagian Dataset': ['90:10', '80:20', '70:30', '60:40'],
                    'Akurasi': [91, 91, 94, 90]}
        df_akurasi_100 = pd.DataFrame(data_100)

        # Membuat layout kolom
        col1, col2 = st.columns(2)

        # Menampilkan chart untuk rasio seleksi fitur 25% di kiri atas
        with col1:
            st.write("""<h6 style = "text-align: center;">Akurasi SVM + QER (Rasio Seleksi Fitur 25%)</h6>""", unsafe_allow_html=True)
            st.bar_chart(df_akurasi_25.set_index('Pembagian Dataset'), height=300)

        # Menampilkan chart untuk rasio seleksi fitur 50% di kanan atas
        with col2:
            st.write("""<h6 style = "text-align: center;">Akurasi SVM + QER (Rasio Seleksi Fitur 50%)</h6>""", unsafe_allow_html=True)
            st.bar_chart(df_akurasi_50.set_index('Pembagian Dataset'), height=300)

        # Menampilkan chart untuk rasio seleksi fitur 75% di kiri bawah
        with col1:
            st.write("""<h6 style = "text-align: center;">Akurasi SVM + QER (Rasio Seleksi Fitur 75%)</h6>""", unsafe_allow_html=True)
            st.bar_chart(df_akurasi_75.set_index('Pembagian Dataset'), height=300)

        # Menampilkan chart untuk rasio seleksi fitur 100% di kanan bawah
        with col2:
            st.write("""<h6 style = "text-align: center;">Akurasi SVM + QER (Rasio Seleksi Fitur 100%)</h6>""", unsafe_allow_html=True)
            st.bar_chart(df_akurasi_100.set_index('Pembagian Dataset'), height=300)

    elif selected == "Implementation":
        text = st.text_area('Masukkan kata yang akan di analisa:')
        submit = st.button("Submit")

        if submit:
            def preprocessing_data(text):
                # Case folding
                text = text.lower()
                text = re.sub("\n", " ", text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                text = re.sub(r'\d+', '', text)

                # Punctual removal
                text = text.translate(str.maketrans("", "", string.punctuation))

                # Tokenization
                text = nltk.word_tokenize(text)

                # Removing stopwords
                stop_words = set(stopwords.words('indonesian'))
                factory = StopWordRemoverFactory()
                stopword_remover = factory.create_stop_word_remover()
                text = [token for token in text if token not in stop_words]
                text = [stopword_remover.remove(token) for token in text]

                # Stemming
                stemmer = PorterStemmer()
                text = [stemmer.stem(token) for token in text]

                # Menggabungkan token kembali menjadi satu string
                return ' '.join(text)

            # Load the preprocessed dataset
            Dt_Ujm = pd.read_csv("dt_stlh_p.csv")
            ulasan_dataset = Dt_Ujm['ulasan']
            sentimen = Dt_Ujm['label']

            # Preprocess the entire dataset
            ulasan_dataset_preprocessed = [preprocessing_data(ulasan) for ulasan in ulasan_dataset]

            # Load the TF-IDF model
            with open('tfidf.pkl', 'rb') as file:
                loaded_data_tfid = pickle.load(file)

            # Transform the entire preprocessed dataset using the TF-IDF model
            tf_idf_baru = loaded_data_tfid.transform(ulasan_dataset_preprocessed)

            # Manual dataset splitting
            total_data = len(ulasan_dataset)
            train_size = int(total_data * 0.8)

            X_train = tf_idf_baru[:train_size]
            y_train = sentimen[:train_size]

            X_test = tf_idf_baru[train_size:]
            y_test = sentimen[train_size:]

            # Train the SVM model
            svm_clf = SVC()
            svm_clf.fit(X_train, y_train)

            # Predictions on the test set
            y_pred = svm_clf.predict(X_test)

            # Compute and display accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f'Akurasi Model: {accuracy * 100:.2f}%')

            # Preprocess user-inputted text and make predictions
            preprocessed_text = preprocessing_data(text)
            v_data = loaded_data_tfid.transform([preprocessed_text])
            y_preds = svm_clf.predict(v_data)

            st.subheader('Prediksi:')
            if y_preds[0] == "positif":
                st.success('Positif')
            else:
                st.error('Negatif')

    elif selected == "Tentang Kami":
        st.write("#####  Skripsi") 
        st.write("Hambali Fitrianto (200411100074)")
