import streamlit as st
import pandas as pd
import pickle
def app():
    option_education = st.sidebar.selectbox(
     'Education? ',
    ('Graduation', 'Master', 'PhD', '2n Cycle', 'Basic'))

    option_marital = st.sidebar.selectbox(
     'Marital Status',
    ('Married', 'Together','Single','Widow','Alone','YOLO','Absurd'))

    option_income = st.sidebar.text_input(
     'Income ?')

    option_kidhome = st.sidebar.text_input(
     'Kid home ? ')

    option_teenhome = st.sidebar.text_input(
     'Teen home ? ')

    option_mnt_wines = st.sidebar.text_input(
     'Wines ? '
    )

    option_mnt_fruit = st.sidebar.text_input(
     'Fruits ? '
    )

    option_mnt_meat = st.sidebar.text_input(
     'Meat ? '
    )

    option_mnt_fish = st.sidebar.text_input(
     'Fish ? '
    )

    option_mnt_sweet = st.sidebar.text_input(
     'Sweet ? '
    )

    option_mnt_gold = st.sidebar.text_input(
     'Gold ? '
    )

    option_deal_purchases = st.sidebar.text_input(
     'Deals Purchases ? '
    )

    option_web_purchases = st.sidebar.text_input(
     'Web Purchases ? '
    )

    option_catalog_purchases = st.sidebar.text_input(
     'Catalog Purchases ? '
    )

    option_store_purchases = st.sidebar.text_input(
     'Store Purchases ? '
    )

    option_web_visitsmonth = st.sidebar.text_input(
     'Web Visits Month ? '
    )

    option_age = st.sidebar.text_input(
     'Age ? '
    )

    option_total_spending = st.sidebar.text_input(
     'Total Spending ? '
    )

    option_children = st.sidebar.text_input(
     'Children ? '
    )

    option_peopleathome = st.sidebar.text_input(
     'People At Home ? '
    )
    option_parent = st.sidebar.text_input(
     'Parent'
    )


    if st.sidebar.button('Predict the Possibility of the Customer to make a purchase'):
        lookup_dict={'Graduation':1, 'Master':1, 'PhD':1, '2n Cycle':0, 'Basic':0}
        lookup_dict1={'Married':0, 'Together':0,'Single':1,'Widow':1,'Alone':1,'YOLO':1,'Absurd':1}


        dict = {'Education':lookup_dict[option_education],
            'Marital_Status':lookup_dict1[option_marital],
            'Income':[option_income],
            "Kidhome":[option_kidhome],
            "Teenhome":[option_teenhome],
            'MntWines':[option_mnt_wines],
            'MntFruits':[option_mnt_fruit],
            "MntMeatProducts":[option_mnt_meat],
            "MntFishProducts":[option_mnt_fish],
            "MntSweetProducts":[option_mnt_sweet],
            "MntGoldProds":[option_mnt_gold],
            "NumDealsPurchases":[option_deal_purchases],
            "NumWebPurchases":[option_web_purchases],
            "NumCatalogPurchases":[option_catalog_purchases],
            "NumStorePurchases":[option_store_purchases],
            "NumWebVisitsMonth":[option_web_visitsmonth],
            "Age":[option_age],
            "TotalSpending":[option_total_spending],
            "Children":[option_children],
            "PeopleAtHome":[option_peopleathome],
            "Parent":[option_parent]
           }
        prediction_df = pd.DataFrame(dict)


        st.write("Customer details for Propensity prediction")
        st.write(prediction_df)

        with open('model.pkl','rb') as r:
            kmp = pickle.load(r)
            
            ypred = kmp.predict(prediction_df)
            st.write("Masuk Kedalam culster : ")
            st.write(ypred)


        
        # with open("model.pkl", 'rb') as pfile:  
        #     propensity_model_loaded=pickle.load(pfile)
        # propensity_model_loaded.predict(prediction_df)
        # # if (y_predicted[0]==1):
        # #     st.write("The customer will order from the website. Probabality of ordering:")
        # #     # st.write(y_predicted[0])
        # # else:
        # #     st.write("The customer will not order from the website. Probabality of ordering:")
        # #     # st.write(y_predicted[0])
        # st.write(propensity_model_loaded.predict_proba(prediction_df))
