from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import regex as re
import json
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# import pickle5 as pickle 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
# from pickle import dump

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
#st.write("### Dosen Pembimbing I: .",unsafe_allow_html=True)
#st.write("### Dosen Pembimbing II: .",unsafe_allow_html=True)

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
        st.write("#### Deskripsi Dataset")
        st.write(""" <p style = "text-align: justify;">Dataset ini berisi ulasan masyarakat terhadap jamu Madura. Selanjutnya, data ulasan ini akan diklasifikasikan ke dalam dua kategori sentimen yaitu positif dan negatif, kemudian dilakukan penerapan algoritma Support Vector Machine dan Seleksi Fitur Query Expansion Ranking untuk mengetahui nilai akurasinya.</p>""", unsafe_allow_html=True)
        st.write("#### Preprocessing Dataset")
        st.write(""" <p style = "text-align: justify;">Preprocessing data merupakan proses untuk mengubah teks yang tidak teratur menjadi lebih terstruktur, yang nantinya akan membantu dalam pengolahan data.</p>""", unsafe_allow_html=True)
        st.write("#### Tahapan Preprocessing Dataset")
        st.write(""" 
        Tahapan preprocessing data melibatkan lima langkah sebagai berikut:
        1. **Case Folding**: Mengubah semua huruf menjadi huruf kecil.
        2. **Punctuation Removal**: Menghapus tanda baca dari teks.
        3. **Tokenizing**: Membagi teks menjadi kata-kata.
        4. **Stopword Removal**: Menghapus kata-kata umum yang tidak memberikan informasi penting.
        5. **Stemming**: Mengubah kata-kata menjadi bentuk dasarnya.

        Di bawah ini adalah contoh dari dataset sebelum dan setelah preprocessing:
        """, unsafe_allow_html=True)

        st.title("Preprocessing Dataset Teks")

        # Load the dataset
        @st.cache
        def load_dataset():
            # Gantilah 'dataset.csv' dengan nama file CSV yang berisi dataset teks
            df = pd.read_csv('dataset.csv')
            return df

        # Preprocessing function
        def preprocess_text(text):
            # Case Folding
            st.write("### Hasil setelah Case Folding:")
            text = text.lower()
            st.write(text)

            # Punctuation Removal
            st.write("### Hasil setelah Punctuation Removal:")
            text = re.sub(r'[^\w\s]', '', text)
            # Hapus tweet khusus
            text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
            text = text.encode('ascii', 'replace').decode('ascii')
            text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
            text = text.replace("http://", " ").replace("https://", " ")
            # Hapus nomor
            text = re.sub(r"\d+", "", text)
            # Hapus tanda baca
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
            # Hapus whitespace
            text = text.strip()
            text = re.sub('\s+', ' ', text)
            # Hapus karakter tunggal
            text = re.sub(r"\b[a-zA-Z]\b", "", text)
            st.write(text)

            # Tokenisasi
            st.write("### Hasil Tokenisasi:")
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)
            st.write(tokens)

            # Stopword Removal
            st.write("### Hasil Setelah Stopword Removal:")
            stop_words = set(stopwords.words('indonesian'))
            filtered_tokens = [word for word in tokens if word not in stop_words]
            st.write(filtered_tokens)

            # Stemming
            st.write("### Hasil Setelah Stemming:")
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
            st.write(stemmed_tokens)

            # Gabungkan token kembali ke dalam teks
            preprocessed_text = ' '.join(stemmed_tokens)
            st.write("### Hasil Akhir Setelah Preprocessing:")
            st.write(preprocessed_text)

            return preprocessed_text

        # Load the dataset
        df = load_dataset()

        # Display the dataset before preprocessing
        st.write("### Dataset Sebelum Preprocessing")
        st.write(df)

        # Preprocess the dataset
        df['preprocessed_text'] = df['ulasan'].apply(preprocess_text)

        # Display the dataset after preprocessing
        st.write("### Dataset Setelah Preprocessing")
        st.write(df[['ulasan', 'preprocessed_text']])


    elif selected == "Implementation":
        #Getting input from user
        word = st.text_area('Masukkan kata yang akan di analisa :')

        submit = st.button("submit")

        if submit:
            def prep_input_data(word, slang_dict):
                #Lowercase data
                lower_case_isi = word.lower()

                #Cleansing dataset
                clean_symbols = re.sub("[^a-zA-Zï ]+"," ", lower_case_isi)

                #Slang word removing
                def replace_slang_words(text):
                    words = nltk.word_tokenize(text.lower())
                    words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
                    for i in range(len(words_filtered)):
                        if words_filtered[i] in slang_dict:
                            words_filtered[i] = slang_dict[words_filtered[i]]
                    return ' '.join(words_filtered)
                slang = replace_slang_words(clean_symbols)

                #Inisialisai fungsi tokenisasi dan stopword
                # stop_factory = StopWordRemoverFactory()
                tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
                tokens = tokenizer.tokenize(slang)

                stop_factory = StopWordRemoverFactory()
                more_stopword = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                                'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                                'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                                '&amp', 'yah']
                data = stop_factory.get_stop_words()+more_stopword
                removed = []
                if tokens not in data:
                    removed.append(tokens)

                #list to string
                gabung =' '.join([str(elem) for elem in removed])

                #Steaming Data
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                stem = stemmer.stem(gabung)
                return lower_case_isi,clean_symbols,slang,gabung,stem
            
            #Kamus
            with open('combined_slang_words.txt') as f:
                data = f.read()
            slang_dict = json.loads(data)

            #Dataset
            Data_ulasan = pd.read_csv("dataprep.csv")
            ulasan_dataset = Data_ulasan['ulasan']
            sentimen = Data_ulasan['label']

            # TfidfVectorizer 
            tfidfvectorizer = TfidfVectorizer(analyzer='word')
            tfidf_wm = tfidfvectorizer.fit_transform(ulasan_dataset)
            tfidf_tokens = tfidfvectorizer.get_feature_names_out()
            df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
            
            #Train test split
            # Memisahkan data menjadi training set dan test set
            X_train, X_test, y_train, y_test = train_test_split(tfidf_wm, sentimen, test_size=0.1, random_state=1)

            # model
            # with open('modelpola.pkl', 'rb') as file:
            #     loaded_model = pickle.load(file)
            # clf = loaded_model.fit(X_train,y_train)
            # y_pred=clf.predict(X_test)

            # Membuat instance objek dari kelas MultinomialNB
            clf = MultinomialNB()

            # Melatih model menggunakan data pelatihan
            clf.fit(X_train, y_train)

            # Membuat prediksi menggunakan data testing
            y_pred = clf.predict(X_test)

            #Evaluasi
            # Menghitung akurasi
            akurasi = accuracy_score(y_test, y_pred)

            # Inputan 
            lower_case_isi,clean_symbols,slang,gabung,stem = prep_input_data(word, slang_dict)
            
            # #Prediksi
            v_data = tfidfvectorizer.transform([stem]).toarray()
            y_preds = clf.predict(v_data)

            st.subheader('Preprocessing')
            st.write(pd.DataFrame([lower_case_isi],columns=["Case Folding"]))
            st.write(pd.DataFrame([clean_symbols],columns=["Cleansing"]))
            st.write(pd.DataFrame([slang],columns=["Slang Word Removing"]))
            st.write(pd.DataFrame([gabung],columns=["Stop Word Removing"]))
            st.write(pd.DataFrame([stem],columns=["Steaming"]))
            # st.write('Case Folding')
            # st.write(lower_case_isi)
            # st.write('Cleaning Simbol')
            # st.write(clean_symbols)
            # st.write('Slang Word Removing')
            # st.write(slang)
            # st.write('Stop Word Removing')
            # st.write(gabung)
            # st.write('Steaming')
            # st.write(stem)

            # st.subheader('Akurasi')
            # st.info(akurasi)
            # Mengubah akurasi menjadi persen dengan 2 desimal
            akurasi_persen = akurasi * 100

            # Menampilkan akurasi
            st.subheader('Akurasi')
            st.info(f"{akurasi_persen:.2f}%")

            

            st.subheader('Prediksi')
            if y_preds == "positive":
                st.success('Positive')
            else:
                st.error('Negative')

            # Classification Report
            classification_rep = classification_report(y_test, y_pred)
            st.subheader('Classification Report:\n')
            st.code(classification_rep)

            # # Confusion Matrix Heatmap
            # st.subheader('Confusion Matrix Heatmap')
            # fig, ax = plt.subplots()
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], ax=ax)
            # st.pyplot(fig)

            # Menghitung confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Menampilkan confusion matrix dengan plot
            st.subheader('Confusion Matrix')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot()

    elif selected == "Tentang Kami":
        st.write("#####  Skripsi") 
        st.write("Hambali Fitrianto (200411100074)")
