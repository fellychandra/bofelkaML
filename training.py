import streamlit as st
import pandas as pd
#import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib
from matplotlib.colors import ListedColormap



st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Machine Learning K-Means')

def app():
    st.subheader('Input dataset dengan tipe xls')
    uploaded_file = st.file_uploader('Choose a dataset',type='xlsx')

    #df=pd.read_csv("marketing_campaign.csv")
    if uploaded_file:
        st.markdown('___')
        df = pd.read_excel(uploaded_file, engine='openpyxl')



        st.subheader('Data Frame')
        st.dataframe(df)

        st.write("Distribution of Orders (Dependent variable)")

        st.write((df["ID"].value_counts()))

        st.write(df.head())

        st.write(df.info())

        st.write("""**DATA CLEANSING**""")

        st.write(df.isna().sum())

        st.write("Menghilangkan Missing Value")
        st.write(df = df.fillna(float(df['Income'].mean())))
        st.write(df['Income'].isna().sum())

        st.write(df.describe())

        st.write( """**DATA EXPLORATION**""")

        st.write(df.info())

        st.write("Menghapus Outliers")
        numerics = ['int64', 'float64']
        numericColumns = df.select_dtypes(include=numerics)
        # print(numericColumns)

        plt.figure(1 , figsize = (45 , 90))
        n = 0
        for x in numericColumns:
            n += 1
            plt.subplot(8 , 8 , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.boxplot(x = df[x])
        st.pyplot(plt.show())

        st.write("""-------------- OKE""")

        df = df[(df["Year_Birth"] > 1939)]
        df = df[(df["Income"]<100000)]
        df = df[(df["MntMeatProducts"]<1500)]
        df = df[(df["MntSweetProducts"]<250)]
        df = df[(df["NumDealsPurchases"]<10)]
        df = df[(df["NumWebPurchases"]<10)]
        df = df[(df["NumCatalogPurchases"]<10)]

        plt.figure(1 , figsize = (45 , 90))
        n = 0
        for x in numericColumns:
            n += 1
            plt.subplot(8 , 8 , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.boxplot(x = df[x])
        st.pyplot(plt.show())


        st.write("Correlation matrix")
        corrmat= df.corr()
        plt.figure(figsize=(20,20))  
        sns.heatmap(corrmat,annot=True, center=0)
        st.pyplot()

        st.write(df.describe())

        st.write("""**FEATURE ENGINEERING**""")

        st.write(print(df["Education"].value_counts()))

        df["Education"] = df["Education"].replace({"Graduation":"Sudah Lulus", "PhD":"Sudah Lulus", "Master":"Sudah Lulus", "2n Cycle":"Belum Lulus", "Basic":"Belum Lulus"})

        st.write((df["Marital_Status"].value_counts()))

        df['Marital_Status'] = df["Marital_Status"].replace({"Married":"Pasangan", "Together":"Pasangan", "Absurd":"Sendiri", "Widow":"Sendiri", "YOLO":"Sendiri", "Divorced":"Sendiri", "Single":"Sendiri", "Alone":"Sendiri"})

        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])

        df["Age"] = (2023-df["Year_Birth"]).round()

        df["TotalSpending"] = df["MntWines"] + df["MntFruits"]+ df["MntMeatProducts"] + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

        df["Children"] = df["Kidhome"] + df["Teenhome"]

        df["PeopleAtHome"] = (df["Marital_Status"].replace({"Sendiri": 1, "Pasangan":2}).astype(int)) + (df["Children"]).astype(int)

        df["Parent"] = np.where(df.Children> 0, 1, 0)

        df_copy = df.copy()

        st.write("Hapus features")
        df_copy = df_copy.drop(["Z_CostContact", "Z_Revenue", "ID", "Year_Birth",'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response', 'Dt_Customer', 'Recency'], axis=1)

        st.write(df_copy.info())

        # Label Encoding
        objectFeature = (df_copy.dtypes == 'object')
        objectFeature = list(objectFeature[objectFeature].index)
        st.write((objectFeature))

        label = LabelEncoder()
        for i in objectFeature:
            df[i] = df[[i]].apply(label.fit_transform)
            df_copy[i] = df_copy[[i]].apply(label.fit_transform)

        # Scaling
        scale = StandardScaler()
        scale.fit(df_copy)
        scaled_ds = pd.DataFrame(scale.transform(df_copy),columns= df_copy.columns)

        st.write("""MODELING""")

        # Menentukan jumlah kluster dengan metode elbow
        inertia = []
        for n in range(1 , 11):
            algorithm = KMeans(n_clusters = n, init='k-means++', n_init = 10 , max_iter=300, random_state= 111)
            algorithm.fit(scaled_ds)
            inertia.append(algorithm.inertia_)
        # Plot elbow
        plt.figure(1 , figsize = (15 ,6))
        plt.plot(np.arange(1 , 11) , inertia , 'o')
        plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
        plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
        plt.show()
        st.pyplot()

        #Initiating the KMeans model 
        fig, ax = plt.subplots()
        km = KMeans(n_clusters = 4, init='k-means++', n_init = 10 , max_iter=300, random_state= 111)
        # fit model and predict clusters
        fitModel = km.fit(scaled_ds)
        predicted = km.predict(scaled_ds)
        #Adding the Clusters feature to the orignal dataframe.
        scaled_ds["Cluster"]= predicted
        df["Cluster"]= predicted
        centroids = km.cluster_centers_

        #Plotting countplot of clusters
        pal = ["#FD8A8A","#F1F7B5", "#A8D1D1","#9EA1D4"]
        pl = sns.countplot(x=scaled_ds["Cluster"], palette= pal)
        
        st.pyplot(fig)

        st.write(scaled_ds.info())

        fig, ax = plt.subplots()
        pl = sns.scatterplot(data = scaled_ds, x = scaled_ds["TotalSpending"], y = scaled_ds["Income"], hue = scaled_ds["Cluster"], palette= pal)
        plt.scatter(
            centroids[:, 17],
            centroids[:, 2],
            s=200,
            linewidths=3,
            color="yellow",
            zorder=10,
        )
        pl.set_title("Cluster's Profile Based On Income And Spending")
        plt.legend()
        plt.show()
        st.pyplot(fig)

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'TotalSpending'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntWines'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntFruits'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntMeatProducts'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntFishProducts'].sum())
        arrayList

        arrayList = []
        for x in range(4) :
            arrayList.append(df.loc[df['Cluster'] == x, 'MntGoldProds'].sum())
        arrayList

        
        df.info()

        info = ['Education','Marital_Status', 'NumDealsPurchases', 'NumWebPurchases', 'NumWebVisitsMonth', 'NumCatalogPurchases','NumStorePurchases', 'Children', 'Parent', 'Age']
        for i in info:
            plt.figure()
            sns.jointplot(x=df[i], y=df["TotalSpending"], hue =df["Cluster"], kind="kde", palette=pal)
            st.pyplot(plt.show())


        plt.figure()
        sns.jointplot(x=df['Parent'], y=df["TotalSpending"], hue =df["Cluster"], kind="kde", palette=pal)
        st.pyplot(plt.show())
        st.write("coba prediksi")
        data = [[15, 39, 11, 6, 7, 0, 1, 3, 4, 5, 6, 1, 2, 3, 4, 6, 7, 7, 8, 20, 21]]
        print(data)
        st.write(km.predict(data))

        st.write(df_copy.head(50))

        df.head()

if __name__=='__main__':
    app()