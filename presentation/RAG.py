# Load required libraries
import boto3
import botocore
import pandas as pd
import numpy as np
import seaborn as sns
# import sklearn
import os
import io
import s3fs
import psycopg2
import urllib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
from botocore.exceptions import ClientError
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_roc_curve, classification_report, accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge

# For naive Bayes
from sklearn.naive_bayes import GaussianNB
# SVM
from sklearn import svm
# import pickle

#from imblearn.over_sampling import SMOTE

import eli5
from eli5.sklearn import PermutationImportance

# Bring in data

# Create donkey variable
df['donkey'] = np.random.normal(0, 1, len(df))

# Classify the feature types based on dtype (as a starting point).
integer_features = list(df.columns[df.dtypes == 'int64'])
cont_features = list(df.columns[df.dtypes == 'float64'])
categorical_features = list(df.columns[df.dtypes == 'object'])


null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum()

drop_rows = list(df[df["area_unit_code"].isnull()][null_columns].index)
df.loc[drop_rows]

# Drop these rows.
df.drop(index = drop_rows, inplace=True)

null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum()

# Set display options
pd.options.display.max_columns = None
pd.options.display.max_rows = None



predictor_columns = ['escalation_qty',
           'consent_category_code',
           'install_distance', 'nzdep2018',
           'ffp_unk',
       'ffp_2000', 'ffp_2012', 'ffp_2013', 'ffp_2014', 'ffp_2015', 'ffp_2016',
       'ffp_2017', 'ffp_2018', 'ffp_2019', 'ffp_2020', 'prod_l2_tbd',
       'prod_l2_nga', 'prod_l2_hyperfibre', 'mdu_class_bldgs',
       'mdu_class_subdivision', 'mdu_class_sdu', 'mdu_class_2_6',
       'mdu_class_7_12', 'mdu_13_32', 'mdu_33_', 'mdu_tbd']
       

label_columns = ['switch_from_green']

df[predictor_columns].dtypes

for var in predictor_columns:
    if var == "donkey":
        continue
    print("Scaling " + var)
    df[var] = df[var].astype("float64")
    
key_columns = ['period_end_wid', 'co_order_no']

info_columns = ['business_dt', 'period_end_wid', 'received_dt', 'tlc', 'sam_no',
       'pipeline_milestone', 'area_unit_code', 'area_unit_name',
       'meshblock_code','consent_category']


#Keep only required predictors.
X = df[predictor_columns]
print("Set size = " + str(len(X)))


# Scale predictors
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=predictor_columns)

X.head()



y = df[label_columns]


# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# Create correlation matrix including all variables
corr = X_train.corr()



fig, ax = plt.subplots(figsize=(14,14))

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    # cmap=sns.diverging_palette(20, 220, n=200),
    # cmap='coolwarm',
    cmap="YlGnBu",
    square=True,
    annot=True,
    cbar=False,
    annot_kws={'size':10},
    fmt=".2f",
    xticklabels=True, 
    yticklabels=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# Fit Random Forest model
# class_weight='balanced_subsample', 

clf = RandomForestClassifier(n_estimators=10, random_state=0, max_depth=None, class_weight='balanced_subsample')
#clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)


# Check it out.
# Note: 0 = 'Green', 1 = 'Red' or 'Amber'
print(classification_report(y_test,y_pred))


# Permutation importance
from sklearn.inspection import permutation_importance
r = permutation_importance(clf, X_test, y_test,
                            n_repeats=5,
                            random_state=0)


for i in r.importances_mean.argsort()[::-1]:
    if True: #r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X.columns[i]} "
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}")


perm = PermutationImportance(clf, random_state=1).fit(X_train, y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist())


Weight	Feature
0.0147 ± 0.0005	install_distance
0.0138 ± 0.0004	nzdep2018
0.0018 ± 0.0004	ffp_2019
0.0010 ± 0.0003	ffp_2016
0.0009 ± 0.0002	ffp_2020
0.0009 ± 0.0002	ffp_2017
0.0008 ± 0.0002	ffp_2015
0.0008 ± 0.0002	consent_category_code
0.0007 ± 0.0003	ffp_2018
0.0006 ± 0.0002	ffp_2012
0.0002 ± 0.0001	ffp_unk
0.0001 ± 0.0000	escalation_qty
0.0000 ± 0.0000	prod_l2_hyperfibre
0 ± 0.0000	ffp_2000
-0.0000 ± 0.0000	mdu_tbd
-0.0004 ± 0.0001	mdu_class_bldgs
-0.0005 ± 0.0003	prod_l2_tbd
-0.0007 ± 0.0002	mdu_class_7_12
-0.0009 ± 0.0003	mdu_class_subdivision
-0.0011 ± 0.0005	prod_l2_nga
… 6 more …









