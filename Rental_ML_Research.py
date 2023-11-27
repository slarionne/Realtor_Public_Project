# data analysis and wrangling

import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from pandas import isnull
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("all_data_1_8.csv")
rental_data = pd.read_csv('full_rental_data_2_19_22.csv')

# cols = ['listing_id']
# merged = pd.merge(data, rental_data, on=cols, how='outer', indicator=True)

"""
From above analysis, I found that there's no overlap between the rental property data and the recently sold data that
I previously downloaded.

"""

#
# corrMatrix = rental_data.corr()
# sns.heatmap(corrMatrix, annot=True)
# plt.show()
#
# rank = corrMatrix['true_or_no_price'].abs().nlargest(30) * 100
# names = rank.index
#
# newdata = rental_data[rental_data['price'].notnull()]
# rental_data['true_or_no_price'] = rental_data['price'].notnull()


newdata = rental_data[rental_data['client_display_flags.is_rental_community'] == False]

"""
Found that homes that had the is_rental_community tagged as true did not show a price, removed as a result
"""


corrMatrix = newdata.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

rank = corrMatrix['price'].abs().nlargest(30) * 100
names = rank.index





data.drop(columns=['Unnamed: 0', 'page_no', 'property_id', 'listing_id', 'rank', 'photo_count', 'branding.state_license'
                   , 'address.country_needed_for_uniq', 'client_display_flags.is_showcase',
                   'client_display_flags.lead_form_phone_required', 'client_display_flags.price_change',
                   'client_display_flags.has_specials', 'client_display_flags.is_MLS_rental',
                   'client_display_flags.is_rental_community', ''],
          inplace=True)

dfcolumns = list(rental_data.columns.values)
print(dfcolumns)

newdata['ZIPcode_rank'] = newdata.groupby("address.postal_code")['price'].rank(method='dense', ascending=True, pct=True)
newdata['NBcode_rank'] = newdata.groupby('address.neighborhoods')['price'].rank(method='dense', ascending=True, pct=True)
newdata['NBname_rank'] = newdata.groupby('address.neighborhood_name')['price'].rank(method='dense', ascending=True, pct=True)


test_data1 = newdata[['address.line','address.neighborhood_name','NBname_rank','ZIPcode_rank','address.postal_code',
                      'price']].sort_values(["address.neighborhood_name", "NBname_rank","address.postal_code",
                                             "ZIPcode_rank"], ascending = (False, False,False,False))


newdata['RentalGrade'] = newdata.groupby(pd.qcut(x=newdata['NBname_rank'],q=3,labels=['low','mid','high']))

newdata['RentalGrade'] = pd.qcut(x=newdata['NBname_rank'],q=3,labels=['Cheap','Average','Expensive'])
newdata = newdata.reset_index(drop=True)
newdata.to_feather('../RealtorAPI/realtor_data_2_20_22.ftr')

"""
- city
- zipcodes
- range ( cheapest, most expensive, median home, average price)
- rental property - ranks (cheapest or most expensive), how long its on the market,
- predict the rental price of the property depending on how neighborhood is predicted



"""