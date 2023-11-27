"""
Realtor_ML
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_absolute_error, log_loss
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('all_data_1_8.csv')

'''

Found that the error for this expanded database of 2,347 rows of data has a slightly worst score but still pretty good
in the grand scheme of things. The average price was $94,737 off from the correct price.


'''



def relevant_columns(data_):
    data_out = data_[['building_size.size', 'baths', 'baths_full', 'beds', 'lot_size.size',
                      'address.postal_code', 'baths_half']].copy()
    data_out['address.postal_code'] = data_out['address.postal_code'].astype('str')
    data_out = data_out.fillna(0.0)
    return data_out


def output_columns(data_):
    data_out = data_['price']
    return data_out


def scale_it(database, column):
    # fit with the Item_MRP
    scales.fit(np.array(database[column]).reshape(-1, 1))
    # transform the data
    database[column] = scales.transform(np.array(database[column]).reshape(-1, 1))
    return database


def quick_analysis(data_):
    check_it = data_.isna().sum()
    check_type = data_.dtypes
    plt.figure(figsize=(10, 7))
    feat_importance_ = pd.Series(RandomForestRegressor.feature_importances_, index=data_.columns)
    feat_importance_.nlargest(7).plot(kind='barh')
    return check_it, check_type

# Data Transformation


y_out = output_columns(data)
data_out = relevant_columns(data)

scales = StandardScaler()
data_out = scale_it(data_out, 'building_size.size')
data_out = scale_it(data_out, 'lot_size.size')

# Data_FS Transformation

data_fs = pd.read_csv('list_for_sale_data_11_16.csv')

y_out_fs = output_columns(data_fs)
data_out_fs = relevant_columns(data_fs)

data_out_fs = scale_it(data_out_fs, 'building_size.size')
data_out_fs = scale_it(data_out_fs, 'lot_size.size')



"""

MACHINE LEARNING

Testing with live data

"""

X_train, X_test, y_train, y_test = train_test_split(data_out, y_out, random_state=35, test_size=0.2)

def score_dataset(X_train, y_train, other):
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(other)
        return preds

predicted_values = score_dataset(X_train, y_train, data_out_fs)

data_fs['newcol'] = predicted_values.tolist()
data_fs.to_csv('predictions_1_8_21.csv')




# from sklearn.pipeline import Pipeline
#
# full_model = Pipeline(
#     steps=[('RandomforReg', classifier_model)],
#     memory=tmpdir)
#
# param_grid = {'classifier__n_neighbors': n_neighbors_list}
# grid_model = GridSearchCV(full_model, param_grid)
# grid_model.fit(X, y)


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid, test_type):
    if test_type == 'RandomForReg':
        model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)
    elif test_type == 'RandomForClass':
        model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=2)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)


score1 = score_dataset(X_train, X_test, y_train, y_test, 'RandomForClass')
score2 = score_dataset(X_train, X_test, y_train, y_test, 'RandomForReg')

log_loss(y, y_hat)

'''

Grid Search

'''



params = DecisionTreeClassifier().get_params()
params = RandomForestRegressor().get_params()




X_train, X_test, y_train, y_test = train_test_split(newdata, output, random_state=35, test_size=0.2)

MAE_scorer = make_scorer(mean_absolute_error)
acc_make_scorer = make_scorer(balanced_accuracy_score)

param_grid = [
    {'max_depth': [2, 3, 5],
     'min_samples_leaf': [1, 2, 3],
     'min_samples_split': [2, 3, 4]}
]

param_grid_MAE = [
    {'n_estimators': [9, 10, 12, 13, 15, 20, 25],
     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7]}
]

grid = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring=acc_make_scorer)
grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid_MAE, scoring=MAE_scorer)

grid.fit(X_train, y_train)
print('Best Hyperparameters: %s ' % grid.best_params_)

# Creating Pipeline


# Define the Pipeline
"""
Step1: get the oultet binary columns
Step2: pre processing
Step3: Train a Random Forest Model
"""
model_pipeline = Pipeline(steps=[('get_outlet_binary_columns', OutletTypeEncoder()),
                                 ('pre_processing', pre_process),
                                 ('random_forest', RandomForestRegressor(max_depth=10, random_state=2))
                                 ])
# fit the pipeline with the training data
model_pipeline.fit(train_x, train_y)

# predict target values on the training data
model_pipeline.predict(train_x)
