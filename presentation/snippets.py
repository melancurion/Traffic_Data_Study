with pd.ExcelWriter('CAS_stats_20220409.xlsx') as writer:  
    df_stats.to_excel(writer, sheet_name='Column stats for CAS data')
    distinct_vals.to_excel(writer, sheet_name='Distinct vals')
    
    
    
One-hot encoding of categorical variables.

# 'ontId' one-hot encoding
ontId_lst = ['__________G-240G-P__','00000G140WC','__________I-240G-R__',\
                         '#','BVL3A5HNAAG010SP','BVL3A8JNAAG010SA','NOCLEICODEXS250WXA',\
                         '00000XS250WXA','NOCLEICODE  BGW320','BVMHZ00ARAU00160CPP',\
                         '','NanoG']

ontId_df = pd.DataFrame(ontId_lst, columns=['ontid'])
dum_df = pd.get_dummies(ontId_df, columns=["ontid"], prefix=["ontid"])
ontId_df = ontId_df.join(dum_df)
ontId_df

# 'link quality' one-hot encoding
linkQual_lst = ['DEGRADED','INTERRUPTED','GOOD','UNKNOWN','ONT_OFF','REDUCED_ROBUSTNESS']

linkQual_df = pd.DataFrame(linkQual_lst, columns=['link_quality'])
dum_df = pd.get_dummies(linkQual_df, columns=["link_quality"], prefix=["link_quality"])
linkQual_df = linkQual_df.join(dum_df)
linkQual_df

# 'link status' one-hot encoding
linkStat_lst = ['NEVER_CONNECTED_ONT','OPER_DOWN','DISCONNECTED',\
                'SWITCHED_OFF','ADMIN_DOWN','CONNECTED','WAITING_LONG','WAITING']

linkStat_df = pd.DataFrame(linkStat_lst, columns=['link_status'])
dum_df = pd.get_dummies(linkStat_df, columns=["link_status"], prefix=["link_status"])
linkStat_df = linkStat_df.join(dum_df)

linkStat_df

# 'onttype' one-hot encoding
ontType_lst = ['GPON','XGS']

ontType_df = pd.DataFrame(ontType_lst, columns=['onttype'])
dum_df = pd.get_dummies(ontType_df, columns=["onttype"], prefix=["onttype"])
ontType_df = ontType_df.join(dum_df)

ontType_df

# 'onttemperaturestate' one-hot encoding
ontTemperatureState_lst = ['NOT_APPLICABLE', 'ABNORMAL', 'NORMAL', 'UNKNOWN']

ontTemperatureState_df = pd.DataFrame(ontTemperatureState_lst, columns=['onttemperaturestate'])
dum_df = pd.get_dummies(ontTemperatureState_df, columns=["onttemperaturestate"], prefix=["onttemperaturestate"])
ontTemperatureState_df = ontTemperatureState_df.join(dum_df)

ontTemperatureState_df



df = pd.merge(df, ontId_df, how='left', on='ontid', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=False,
         suffixes=('_x', '_y'), copy=True, indicator=False).drop('ontid', axis=1)
         
df = pd.merge(df, linkQual_df, how='left', on='link_quality', left_on=None, right_on=None,
         left_index=False, right_index=False, sort=False,
         suffixes=('_x', '_y'), copy=True, indicator=False).drop('link_quality', axis=1)
         
...


null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum()


drop_rows = list(df[df["ontid_"].isnull()][null_columns].index)


# Scale the continuous features
scaler = StandardScaler()
df[cont_features] = scaler.fit_transform(df[cont_features])


# Save this to CSV to permit easier rework.
df.to_csv("s3://chorus-analytics-swamp/Darryl_Boswell/Transfer/prepared_encoded_majority_train_undersample_2_5/")

# Classify the feature types based on dtype (as a starting point).
integer_features = list(df.columns[df.dtypes == 'int64'])
cont_features = list(df.columns[df.dtypes == 'float64'])
categorical_features = list(df.columns[df.dtypes == 'object'])


# Impute missing values to avoid value errors in model.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df[cont_features])
df[cont_features] = imp.transform(df[cont_features])

# Split into predictors matrix and outcomes vector
X_train = df_train.drop(['target_truck_roll', 'tranche'], axis=1)
y_train = df_train[['target_truck_roll']]
X_test = df_test.drop(['target_truck_roll', 'tranche'], axis=1)
y_test = df_test[['target_truck_roll']]
X_val = df_val.drop(['target_truck_roll', 'tranche'], axis=1)
y_val = df_val[['target_truck_roll']]


#Create key frames for each case.
X_train_key = pd.DataFrame(X_train, columns=['address', 'readtime'])
X_test_key = pd.DataFrame(X_test, columns=['address', 'readtime'])
X_val_key = pd.DataFrame(X_val, columns=['address', 'readtime'])


keep_columns = ['rxopticalsignallevel_avg', 'rxopticalsignallevel_max',
       'rxopticalsignallevel_min',
       'rxopticalsignallevel_stddeviation', 'txopticalsignallevel_avg',
       'txopticalsignallevel_max', 'txopticalsignallevel_min',
       'txopticalsignallevel_stddeviation', 'rxopticalsignallevelatolt_avg',
       'rxopticalsignallevelatolt_max', 'rxopticalsignallevelatolt_min',
       'rxopticalsignallevelatolt_notnullvaluesnr',
       'rxopticalsignallevelatolt_stddeviation',
       'rfopticalsignallevel_avg', 'rfopticalsignallevel_max',
       'rfopticalsignallevel_min',
       'rfopticalsignallevel_stddeviation',
       'rxopticalsignallevel_avg_mm', 'rxopticalsignallevel_min_mm',
       'rxopticalsignallevel_max_mm', 'txopticalsignallevel_avg_mm',
       'txopticalsignallevel_min_mm', 'txopticalsignallevel_max_mm',
       'mtbe',
       'link_quality_DEGRADED', 'link_quality_GOOD',
       'link_quality_INTERRUPTED', 'link_quality_ONT_OFF',
       'link_quality_REDUCED_ROBUSTNESS', 'link_quality_UNKNOWN',
       'link_status_ADMIN_DOWN', 'link_status_CONNECTED',
       'link_status_DISCONNECTED', 'link_status_NEVER_CONNECTED_ONT',
       'link_status_OPER_DOWN', 'link_status_SWITCHED_OFF',
       'link_status_WAITING', 'link_status_WAITING_LONG',
       'ontid_#',
       'ontid_00000G140WC',
       'ontid_00000XS250WXA',
       'ontid_BVL3A5HNAAG010SP',
       'ontid_BVL3A8JNAAG010SA',
       'ontid_BVMHZ00ARAU00160CPP',
       'ontid_NOCLEICODE  BGW320',
       'ontid_NOCLEICODEXS250WXA',
       'ontid_NanoG',
       'ontid___________G-240G-P__',
       'ontid___________I-240G-R__',
       'onttype_GPON','onttype_XGS',
       'onttemperaturestate_ABNORMAL', 'onttemperaturestate_NORMAL',
       'onttemperaturestate_NOT_APPLICABLE', 'onttemperaturestate_UNKNOWN',
       'donkey']

#Keep only required predictors.
X_train = X_train[keep_columns]
X_test = X_test[keep_columns]
X_val = X_val[keep_columns]


print("Training set size = " + str(len(X_train)))
print("Test set size = " + str(len(X_test)))
print("Validation set size = " + str(len(X_val)))


# Fit Random Forest model
# class_weight='balanced_subsample', 

clf = RandomForestClassifier(n_estimators=20, random_state=0, max_depth=10, max_features=len(X_train.columns))
clf.fit(X_train, y_train.values.ravel())


y_pred = clf.predict(X_test)



print("Number of cases where prediction is 0: " + str(sum(y_pred == 0)))
print("Number of cases where prediction is 1: " + str(sum(y_pred == 1)))
Number of cases where prediction is 0: 454352
Number of cases where prediction is 1: 43910
#y_probs = clf.predict_proba(X_test)
#y_probs = pd.DataFrame(y_probs, columns=['Prob_0','Prob_1'])
# Check it out.
print(classification_report(y_test,y_pred))
              precision    recall  f1-score   support

           0       0.87      0.94      0.90    417499
           1       0.46      0.25      0.33     80763

    accuracy                           0.83    498262
   macro avg       0.67      0.60      0.62    498262
weighted avg       0.80      0.83      0.81    498262


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
print("---------------")

for f in range(X_train.shape[1]):
    print("%d. %s\t(%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))


# Plot the forest's impurity-based feature importances
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), labels=X_train.columns[indices])
#plt.xticks(range(X_train.shape[1]), indices)
plt.xticks(rotation=90, ha='right')
plt.xlim([-1, X_train.shape[1]])

plt.show()


# Make predictions for validation set, and create probabilities matrix.
y_val_pred = clf.predict(X_val)
y_val_pred_probs = clf.predict_proba(X_val)
#y_val_pred_probs = pd.DataFrame(y_val_pred_probs,)
#New_Labels = ['Prob_0', 'Prob_1']
#y_val_pred_probs.columns = New_Labels
# Check it out.
print(classification_report(y_val,y_val_pred))
# Check it out.
print(classification_report(y_val,y_val_pred))
              precision    recall  f1-score   support

           0       0.87      0.94      0.90    417463
           1       0.46      0.25      0.32     80734

    accuracy                           0.83    498197
   macro avg       0.66      0.60      0.61    498197
weighted avg       0.80      0.83      0.81    498197






