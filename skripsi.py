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
from sklearn.model_selection import train_test_split
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
<center><h2 style = "text-align: justify;">ANALISIS SENTIMEN JAMU MADURA MENGGUNAKAN ALGORITMA SUPPORT VECTOR MACHINE DAN QUERY EXPANSION RANKING</h2></center>
""",unsafe_allow_html=True)
#st.write("### Dosen Pembimbing I: Dr. Rika Yunitarini, ST., MT.",unsafe_allow_html=True)
#st.write("### Dosen Pembimbing II: Fifin Ayu Mufarroha, S.Kom., M.Kom.",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://asset.kompas.com/crops/78bBP1gjXGFghLuRKY-TrLpD7UI=/0x0:1000x667/750x500/data/photo/2020/09/19/5f660d3e0141f.jpg" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Home","Dataset", "Implementation", "Tentang Kami"], 
            icons=['house', 'bar-chart', 'check2-square', 'person'], menu_icon="cast", default_index=0,
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

                return text

            Dt_Ujm = pd.read_csv("dt_stlh_p.csv")
            ulasan_dataset = Dt_Ujm['ulasan']
            sentimen = Dt_Ujm['label']

            ulasan_dataset_preprocessed = [preprocessing_data(ulasan) for ulasan in ulasan_dataset]

            with open('tfidf.pkl', 'rb') as file:
                loaded_data_tfid = pickle.load(file)
            tf_idf_baru = loaded_data_tfid.fit_transform(ulasan_dataset_preprocessed)

            X_train, X_test, y_train, y_test = train_test_split(tf_idf_baru, sentimen, test_size=0.1, random_state=42)

            svm_clf = SVC(kernel='linear')
            svm_clf.fit(X_train, y_train)

            y_pred = svm_clf.predict(X_test)

            st.subheader('Text Analysis with SVM')
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
