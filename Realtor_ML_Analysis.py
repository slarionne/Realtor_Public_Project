# data analysis and wrangling

import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
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

# corrMatrix = data.corr()
# sns.heatmap(corrMatrix, annot=True)
# plt.show()

# rank = corrMatrix['price'].abs().nlargest(30) * 100
# names = rank.index

data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'property_id', 'listing_id', 'prop_type', 'list_date', 'last_update',
                   'prop_status', 'sold_history', 'products', 'agents', 'rdc_web_url', 'rdc_app_url',
                   'data_source_name',
                   'page_no', 'list_tracking', 'photo_count', 'photos', 'address.line', 'address.street_number',
                   'address.street', 'address.street_suffix', 'address.unit', 'address.city', 'address.state_code',
                   'address.lat', 'address.county', 'address.neighborhood_name', 'address.neighborhoods',
                   'address.lon', 'mls.abbreviation', 'mls.id', 'client_display_flags.presentation_status',
                   'client_display_flags.is_showcase', 'client_display_flags.lead_form_phone_required',
                   'client_display_flags.price_change', 'client_display_flags.is_co_broke_email',
                   'client_display_flags.has_open_house', 'client_display_flags.is_foreclosure',
                   'client_display_flags.is_co_broke_phone', 'client_display_flags.is_new_listing',
                   'client_display_flags.is_new_plan', 'client_display_flags.is_turbo',
                   'client_display_flags.is_office_standard_listing', 'client_display_flags.suppress_map_pin',
                   'client_display_flags.show_contact_a_lender_in_lead_form',
                   'client_display_flags.show_veterans_united_in_lead_form',
                   'client_display_flags.is_showcase_choice_enabled', 'client_display_flags.is_recently_sold',
                   'office.advertiser_id', 'office.name', 'office.email', 'office.href', 'office.photo', 'office.id',
                   'office.phones', 'lot_size.units', 'building_size.units', 'price_reduced_date', 'office.photo.href',
                   'rank', 'client_display_flags.is_new_construction', 'client_display_flags.is_short_sale'],
          inplace=True)

# dfcolumns = list(data.columns.values)

# datahead = data.head()
# datatail = data.tail()
#
# data.info()
# descdata = data.describe()

# data.sort_values('address.postal_code', inplace=True)

# Figuring out what the postal codes look like

# lib = data['address.postal_code'].unique()

# Ranks home within zipcode by price without any added analysis. (rank is between 0 and 1)

data['ZIPcode_rank'] = data.groupby("address.postal_code")['price'].rank(method='dense', ascending=True, pct=True)

"""
Cleaning the Data
"""

# Bathroom Total (new feature)
data['TotalBath'] = (data['baths_full'] + (0.5 * data['baths_half']))
data.drop(columns=['baths_half', 'baths_full', 'baths'], inplace=True)

# Year Built quick_analysis
# new = data[data['year_built'] > 2014]
# newother = data[(data['year_built'] < 2014) & (data['is_new_construction'] == True)]
# new2 = new[['year_built', 'is_new_construction', 'address.line', 'price']]
# newother2 = newother[['year_built', 'is_new_construction', 'address.line', 'price']]

'''
After looking up the homes, found that anything built around this time or with new construction will look new

Also from the analysis performed below and printed in year_band_analysis1, found that the newer homes are 
obviously worth more. (1952-2020) and between homes built from (1925-1937) are worth more. May have to consider
a deeper breakdown by perhaps making more cuts in the data (currently only 4).

Performed similar analysis on (new construction) homes, found the same sort of increase which means that these 
2 variables can be correlated to each other. New construction occurred on homes between 1925-1937 for the most part.
'''

data['YearBand'] = pd.qcut(data['year_built'], 4)
data[['YearBand', 'ZIPcode_rank']].groupby(['YearBand'], as_index=False).mean().sort_values(by='YearBand',
                                                                                            ascending=True)

"""
Found that longitude and latitude correlates with price as well. May consider performing another analysis at 
a future time to can determine where the cut off point is for price differences or maybe even a machine LEARNING
analysis on how this can be performed automatically in any town or city when enough data is gathered. Will
continue with current analysis for now just to get it done.

"""


# creating new column for homes that look new


def conditions(s):
    if (s['year_built'] > 2014) or (s['year_built'] < 2014 & s['is_new_construction'] == True):
        return 1
    else:
        return 0


data['updated_or_new'] = data.apply(conditions, axis=1)

data.drop(columns=['year_built', 'is_new_construction', 'YearBand'], inplace=True)

# Filling in missing values


# Creating bedtable dictionary based on bed pricebands


data['PriceBand'] = pd.qcut(data['price'], 6)
bedtable = data[['PriceBand', 'beds']].groupby(['PriceBand'], as_index=False).mean().sort_values(by='PriceBand',
                                                                                                 ascending=True)

bedtable['beds'] = bedtable['beds'].apply(np.floor)

bedtabledict = bedtable.set_index('PriceBand').beds.to_dict()

# Run to Fill nan in beds column

data['beds'] = data['beds'].fillna(data['price'].map(bedtabledict))

data['garage'] = data['garage'].fillna(0)

"""
Continue fill nan: below take from kaggle written code
"""

# data.info()

print(f'Only features contained missing value in database')
temp = data.isnull().sum()
print(temp.loc[temp != 0], '\n')
#
# # For features having smaller than 100 missing values
# null_100 = data.columns[list((temp < 100) & (temp != 0))]
# num = data[null_100].select_dtypes(include=np.number).columns
# non_num = data[null_100].select_dtypes(include='object').columns
#
# # Numerous features
# data[num] = data[num].apply(lambda x: x.fillna(x.median()))
#
# # Object features
# data[non_num] = data[non_num].apply(lambda x: x.fillna(x.value_counts().index[0]))

"""
Back to my work
"""


# Filling all columns with same steps as Beds


def fill_in_logical(data_, column_name):
    # filling in data in existing dataframe with relevant section values, returns temptables for reference
    data_['CohortBand'] = pd.qcut(data_['price'], 6)
    temptables = data_[['CohortBand', column_name]].groupby(['CohortBand'], as_index=False).mean().sort_values(
        by='CohortBand', ascending=True)
    temptables[column_name] = temptables[column_name].apply(np.floor)
    temptabledict = temptables.set_index('CohortBand')[column_name].to_dict()
    data_[column_name] = data_[column_name].fillna(data_['price'].map(temptabledict))
    return temptables


fill_in_logical(data, 'lot_size.size')
fill_in_logical(data, 'building_size.size')
fill_in_logical(data, 'TotalBath')

data.drop(columns=['PriceBand', 'CohortBand'], inplace=True)
data.drop(columns=['address.state'], inplace=True)

"""

The Following is taken from a script found on kaggle --->

2.4.1 Highly skewed numeric features
Highly skewed numeric features are the heavy-tail features like our target features
We decide whether a feature is skewness or not based on the value of "skewness" statistics measurement
All skewed features will be normalize by Box-cox normalization technique


"""

# # Normalize skewness feature using Log function
# skew_features = df_all[num_features].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
# skew_features = skew_features[abs(skew_features) > 0.75]
# print(skew_features)
#
# # Apply Box cox for skewness > 0.75
# for feat in skew_features.index:
#     df_all[feat] = np.log1p(df_all[feat])


"""
End of transformation
"""

import pickle

# Dump it(save it in binary format)
with open('ML_data_10_5.pickle', 'wb') as ML_data_file:
    pickle.dump(data, ML_data_file)

# Load the data - No need to do Feature Engineering again
with open('ML_data_10_5.pickle', 'rb') as ML_data_file:
    data = pickle.load(ML_data_file)

# Continue with your modeling


# # Import sklearn everythings
# import time
#
#
# def timer(f):
#     start = time.time()
#     res = f()
#     end = time.time()
#     print("fitting: {}".format(end - start))
#     return res
#
#
# def build_model_for_data(data, target):
#     X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2)
#     pipeline = male_pipeline(LinearRegression())
#     model = timer(lambda: pipeline.fit(X_train, y_train))
#     return (X_test, y_test, model)


"""
Brainstorming ways to transform data, more specifically the zipcode column. I will attempt to simply encode or 
dummy the column, adding a 1 for each zipcode and increasing the feature least by alot. I'm also considering
creating a new categorical column feature where I break the zipcodes up by 3 different types. Poor, Middle, Rich 
class. This feels a little blunt but is the way we look at the the neighborhoods currently. I'll try to first way,
then revisit this second idea based on the resulting score.

"""

data['address.postal_code'] = data['address.postal_code'].astype('str')

data = pd.get_dummies(data)

# Label encoding instead - See below description for why

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(data['address.postal_code'])

# le.classes_

data_new = data.copy()

data_new['address.postal_code'] = le.transform(data_new['address.postal_code'])

# Reversing the transformation performed on the data
le.inverse_transform(data_new['address.postal_code'])

"""
Found that get dummies or one hot encoding can cause problems for random forest algorithms, so I'm working on 
perfomring label encoding and will check how this affects the output. I will also consider getting rid of the 
column outright or converting the postal_code column to a column that categorizes by wealth class. 

"""

X = data_new.drop(["ZIPcode_rank"], axis=1)
y = data_new["ZIPcode_rank"]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Now create pipline
pipeline_RFR = Pipeline([('pca1', PCA(n_components=2)),
                         ('rfr_classifier', RandomForestRegressor())])

# Now create pipline
pipeline_SVR = Pipeline([('scaler2', StandardScaler()),
                         ('pca2', PCA(n_components=2)),
                         ('svr_classifier', SVR(kernel='rbf'))])

pipeline_RFR2 = Pipeline([('rfr_classifier2', RandomForestRegressor())])

# Lets make the list of pipelines

pipelines = [pipeline_RFR, pipeline_SVR, pipeline_RFR2]

best_accuracy = 0.0
best_classifier = 0
best_pipeline = ""

# Dictionary of pipelines and classifier types for easy of reference

pipe_dict = {0: 'Random Forest Regression w PCA', 1: 'Support Vector Regression', 2: '* Random Forest Regression'}

# Fit the pipeline_SVR
for pipe in pipelines:
    pipe.fit(X_train, y_train)

for i, model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i], round(model.score(X_test, y_test) * 100, 2)), "%")

"""
Testing some model Ideas

"""

from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
Y_pred = svr.predict(X_test)
acc_svr = round(svr.score(X_train, y_train) * 100, 2)
acc_svr

randomforest = RandomForestRegressor()
randomforest.fit(X_train, y_train)
Y_pred = randomforest.predict(X_test)
acc_randomforest = round(randomforest.score(X_train, y_train) * 100, 2)
acc_randomforest



"""
Using highest accuracy level rated model to predict model

"""

prediction_test = pipeline_RFR2.predict(X)
X_copy = X.copy()
X_copy['predictions'] = prediction_test.tolist()
X_copy["official_answer"] = y
X_copy.to_csv("first_test_10_5.csv")

"""
Pickle machine learning model
"""

#import module
# import pickle
#
# #Dump the model
# with open('RandomForestR1.pickle','wb') as modelFile:
#      pickle.dump(pipeline_RFR,modelFile)


#import module
import pickle
#Load the model - No need to TRAIN it again(6 hours saved)
with open('RandomForestR2.pickle','rb') as modelFile:
     model = pickle.load(modelFile)
#Predict with the test set
prediction = model.predict(X_test)


"""
Create pipeline for transforming and running algorithm
"""

