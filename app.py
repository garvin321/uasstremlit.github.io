import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import altair as alt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



st.set_page_config(page_title='UAS PENAMBANGAN DATA')
st.markdown("<h1 style='text-align: center;'>UAS PENAMBANGAN DATA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Garvin TF|20141110061</p>", unsafe_allow_html=True)
st.write("---")







home, preprocessing, modeling, implementasi = st.tabs(["Home", "Preprocessing", "Modeling", "Implementasi"])

voice=pd.read_csv('voice.csv')
voice.head()


with home:
    st.write("# Deskripsi Dataset ")
    st.write("#### Dataset yang digunakan adalah dataset pengenalan gender menggunakan suara, dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('voice.csv')
    st.dataframe(df)
    st.write("###### Resource Dataset : https://www.kaggle.com/datasets/primaryobjects/voicegender")
    st.write(" Dataset ini berisikan 3000 data frekuensi suara manusia dengan range frekuensi 0hz-280hz yang di ambil dari perempuan maupun laki laki . ")
    
with preprocessing:
    st.title(" Normalisasi Data")
    st.write("""###### Rumus Normalisasi Data :""")
    st.image('rumus_normalisasi.png', use_column_width=False, width=250)

    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    # Mendefinisikan Varible X dan Y
    X = df.drop(columns=['label'])
    y = df['label'].values
    df
    st.subheader("Pemisahan Kolom Result Sebagai Atribut Target")
    X
    df_min = X.min()
    df_max = X.max()

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.label).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]]
    })

    st.write(labels)

with modeling:
    # Nilai X training dan Nilai X testing
    training, test = train_test_split(
        scaled_features, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(
        y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        mlp_model = st.checkbox('ANNBackpropagation')

        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        # Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        # KNN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))
        # Menggunakan 2 layer tersembunyi dengan 100 neuron masing-masing
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_predict))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(
                    gaussian_akurasi))
            if k_nn:
                st.write(
                    "Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree:
                st.write(
                    "Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if mlp_model:
                st.write(
                    'Model ANN (MLP) accuracy score: {0:0.2f}'.format(mlp_accuracy))
with implementasi:                
    with st.form("my_form"):
            st.subheader("Implementasi")
            
            meanfreq = st.number_input('Masukkan mean frequency : ')
            sd = st.number_input('masukan standard deviation of frequency : ')
            median = st.number_input('Masukkan median frequency (in kHz) : ')
            Q25 = st.number_input('Masukkan first quantile (in kHz) : ')
            Q75 = st.number_input('Masukkan third quantile (in kHz) : ')
            IQR = st.number_input('Masukkan interquantile range (in kHz) : ')
            skew = st.number_input('Masukkan skewness : ')
            kurt = st.number_input('Masukkan kurtosis : ')
            spent = st.number_input('Masukkan spectral entropy : ')
            sfm = st.number_input('Masukkan spectral flatness : ')
            mode = st.number_input('Masukkan mode frequency : ')
            centroid = st.number_input('Masukkan frequency centroid : ')
            peakf = st.number_input('Masukkan peak frequency : ')
            meanfun = st.number_input('Masukkan average of fundamental frequency measured across acoustic signal : ')
            minfun = st.number_input('Masukkan minimum of fundamental frequency measured across acoustic signal : ')
            maxfun = st.number_input('Masukkan maximum of fundamental frequency measured across acoustic signal : ')
            meandom = st.number_input('Masukkan average of dominant frequency measured across acoustic signal : ')
            mindom = st.number_input('Masukkan minimum of dominant frequency measured across acoustic signal : ')
            maxdom = st.number_input('Masukkan maximum of dominant frequency measured across acoustic signal : ')
            dfrange = st.number_input('Masukkan range of dominant frequency measured across acoustic signal : ')
            modindx = st.number_input('Masukkan modulation index : ')
            model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi dibawah ini:',
                                 ('Naive Bayes', 'K-NN', 'Decision Tree', 'ANNBackpropaganation'))
            apply_pca = st.checkbox("Include PCA")
            prediksi = st.form_submit_button("Submit")

            if prediksi:
                inputs = np.array([
                    meanfreq,
                    sd,
                    median,
                    Q25,
                    Q75,
                    IQR,
                    skew,
                    kurt,
                    spent,
                    sfm,
                    mode,
                    centroid,
                    peakf,
                    meanfun,
                    minfun,
                    maxfun,
                    meandom,
                    mindom,
                    maxdom,
                    dfrange,
                    modindx,
                ])

                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                # if apply_pca:
                #     pca = PCA(n_components=2)
                #     X_pca = pca.fit_transform(X)
                #     input_norm = pca.fit_transform(input_norm)

                if apply_pca and X.shape[1] > 1 and X.shape[0] > 1:
                    pca = PCA(n_components=min(X.shape[1], X.shape[0]))
                    X_pca = pca.fit_transform(X)
                    input_norm = pca.transform(input_norm)

                if model == 'Naive Bayes':
                    mod = gaussian
                    if apply_pca:
                        input_norm = pca.transform(input_norm)
                if model == 'K-NN':
                    mod = knn
                    if apply_pca:
                        input_norm = pca.transform(input_norm)
                if model == 'Decision Tree':
                    mod = dt
                    if apply_pca:
                        input_norm = pca.transform(input_norm)
                if model == 'ANNBackpropaganation':
                    mod = mlp
                    if apply_pca:
                        input_norm = pca.transform(input_norm)

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :', model)

                st.write(input_pred)
                ada = 1
                tidak_ada = 0
                if input_pred == ada:
                    st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                             model, 'Ini Adalah Laki-Laki')
                else:
                    st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                             model, 'Ini Adalah Perempuan')
    


            







    
