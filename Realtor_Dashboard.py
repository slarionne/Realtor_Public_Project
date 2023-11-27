# Import necessary libraries
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# App Title and Description
st.title('Real Estate Analysis')

st.markdown("""
This app performs simple extract of Real Estate data for Washington DC!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Rapid-API: Real Estate](https://rapidapi.com/apidojo/api/realty-in-us/).
""")

# Sidebar for User Input Features
st.sidebar.header('User Input Features')

# Load Data from Feather file
df = pd.read_feather("realtor_data_2_20_22.ftr")

# Dropdown for Neighborhood Selection
list_hood = df['address.neighborhood_name'].dropna().unique().tolist()
list_hood.sort()
selected_hood = st.sidebar.selectbox('Neighborhood', list_hood)

# Function to Load Data based on Neighborhood
@st.cache
def load_data(hood):
    df2 = df[['price', 'address.line', 'address.neighborhood_name', 'NBname_rank', 'ZIPcode_rank', 'RentalGrade',
              'address.postal_code']]
    msk = df2['address.neighborhood_name'] == hood
    df2return = df2[msk]
    return df2return

# Load Data for the Selected Neighborhood
prop_stats = load_data(selected_hood)

# Sidebar for Team Selection
sorted_unique_team = sorted(prop_stats.NBname_rank.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Display Real Estate Details for Selected Neighborhood(s)
st.header('Display real estate details for Selected Neighborhood(s)')
st.dataframe(prop_stats)

# Additional Visualization (Commented out)
# df2 = df[['address.line','address.neighborhood_name','NBname_rank','ZIPcode_rank','RentalGrade','address.postal_code',
#                       'price']].sort_values(["address.neighborhood_name", "NBname_rank","address.postal_code",
#                                              "ZIPcode_rank"], ascending = (False, False,False,False))
# df2.info()
# msk = df2['address.neighborhood_name'] == 'Woodley Park'
# dftest = df[msk]
# dftest = df2[["NBname_rank",'address.postal_code']]
# fig = plt.scatter(dftest, x = 'address.postal_code', y = 'NBname_rank', color = 'address.postal_code')
# fig.show()
# sns.relplot(x = 'address.neighborhood_name', y= 'price', hue = 'RentalGrade',data = df, kind = 'line')
# sns.catplot(x = 'address.neighborhood_name', y= 'price', data = df, kind = 'violin')