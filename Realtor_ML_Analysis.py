# Data Analysis and Wrangling

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load Data
data = pd.read_csv("all_data_1_8.csv")

# Drop Unnecessary Columns
columns_to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'property_id', 'listing_id', 'prop_type', 'list_date', 'last_update',
                   'prop_status', 'sold_history', 'products', 'agents', 'rdc_web_url', 'rdc_app_url', 'data_source_name',
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
                   'rank', 'client_display_flags.is_new_construction', 'client_display_flags.is_short_sale']

data.drop(columns=columns_to_drop, inplace=True)

# Feature Engineering
data['TotalBath'] = (data['baths_full'] + (0.5 * data['baths_half']))
data.drop(columns=['baths_half', 'baths_full', 'baths'], inplace=True)

data['ZIPcode_rank'] = data.groupby("address.postal_code")['price'].rank(method='dense', ascending=True, pct=True)

# Additional Data Cleaning and Filling Missing Values
# ... (skipped for brevity)

# Label Encoding for 'address.postal_code'
data['address.postal_code'] = data['address.postal_code'].astype('str')
le = preprocessing.LabelEncoder()
le.fit(data['address.postal_code'])
data['address.postal_code'] = le.transform(data['address.postal_code'])

# Machine Learning Model

X = data.drop(["ZIPcode_rank"], axis=1)
y = data["ZIPcode_rank"]

# Standardize and Split Data
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create Pipelines
pipeline_RFR = Pipeline([('pca1', PCA(n_components=2)),
                         ('rfr_classifier', RandomForestRegressor())])

pipeline_SVR = Pipeline([('scaler2', StandardScaler()),
                         ('pca2', PCA(n_components=2)),
                         ('svr_classifier', SVR(kernel='rbf'))])

pipeline_RFR2 = Pipeline([('rfr_classifier2', RandomForestRegressor())])

pipelines = [pipeline_RFR, pipeline_SVR, pipeline_RFR2]

# Model Evaluation
for i, model in enumerate(pipelines):
    print("{} Test Accuracy: {}%".format(pipe_dict[i], round(model.score(X_test, y_test) * 100, 2)))

# Save Model
with open('RandomForestR2.pickle', 'wb') as modelFile:
    pickle.dump(pipeline_RFR, modelFile)

# Load Model
with open('RandomForestR2.pickle', 'rb') as modelFile:
    model = pickle.load(modelFile)

# Predict with the Test Set
prediction = model.predict(X_test)