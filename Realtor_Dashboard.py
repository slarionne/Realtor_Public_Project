import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('Real Estate Analysis')

st.markdown("""
This app performs simple extract of Real Estate data for Washington DC!
* **Python libraries:** base64, pandas, streamlit
* **Data source:** [Rapid-API: Real Estate](https://rapidapi.com/apidojo/api/realty-in-us/).
""")

st.sidebar.header('User Input Features')

df = pd.read_feather("realtor_data_2_20_22.ftr")

list_hood = df['address.neighborhood_name'].dropna().unique().tolist()
list_hood.sort()

selected_hood = st.sidebar.selectbox('Neighborhood', list_hood)


@st.cache
def load_data(hood):
    df2 = df[['price', 'address.line', 'address.neighborhood_name', 'NBname_rank', 'ZIPcode_rank', 'RentalGrade',
              'address.postal_code']]
    msk = df2['address.neighborhood_name'] == hood
    df2return = df2[msk]
    return df2return

prop_stats = load_data(selected_hood)

# Sidebar - Team selection
sorted_unique_team = sorted(prop_stats.NBname_rank.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)


st.header('Display real estate details for Selected Neighborhood(s)')
# st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
# test = prop_stats.astype(str)
st.dataframe(prop_stats)






# df2 = df[['address.line','address.neighborhood_name','NBname_rank','ZIPcode_rank','RentalGrade','address.postal_code',
#                       'price']].sort_values(["address.neighborhood_name", "NBname_rank","address.postal_code",
#                                              "ZIPcode_rank"], ascending = (False, False,False,False))
# df2.info()

#
#
# msk = df2['address.neighborhood_name'] == 'Woodley Park'
# dftest = df[msk]
# dftest = df2[["NBname_rank",'address.postal_code']]


# fig = plt.scatter(dftest, x = 'address.postal_code', y = 'NBname_rank', color = 'address.postal_code')
# fig.show()
#
#
# import plotly.io as pio
# pio.renderers.default = "browser"

# sns.relplot(x = 'address.neighborhood_name', y= 'price', hue = 'RentalGrade',data = df, kind = 'line')
# sns.catplot(x = 'address.neighborhood_name', y= 'price', data = df, kind = 'violin')
