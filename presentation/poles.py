import graphviz

import pydot


# Create a set of binary cluster variables to distinguish respectively between each cluster and all other clusters.

# We have converted the problem into a binary classification problem. What is left is to train a classifier 
# and use its feature_importances_ method implemented in scikit-learn to get the features that have the most 
# discriminatory power between all clusters and the targeted cluster. We also need to map them to their 
# feature names sorted by the weights.

print(df['Cluster'].value_counts())
df['Binary Cluster 0'] = df['Cluster'].map({0:1, 1:0, 2:0, 3:0})
df['Binary Cluster 1'] = df['Cluster'].map({0:0, 1:1, 2:0, 3:0})
df['Binary Cluster 2'] = df['Cluster'].map({0:0, 1:0, 2:1, 3:0})
df['Binary Cluster 3'] = df['Cluster'].map({0:0, 1:0, 2:0, 3:1})
#print("\n", df["Binary Cluster 0"].value_counts())

# Drop the Cluster variable
df.drop("Cluster", axis=1, inplace=True)

def get_classifier(data, feature_list, cluster_binary_target, n_estimators = 100, max_depth=6):
    # Train a classifier
    #clf = DecisionTreeClassifier()
    clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=None, splitter='best')
    #clf = RandomForestClassifier(n_estimators, max_depth=max_depth, n_jobs=1, verbose=0, random_state=1)
    return clf.fit(data[feature_list].values, cluster_binary_target.values)  
    
# Create reverse_scaling function for age.
#def reverse_scale_age(age):
#    return scaler.inverse_transform([age])



def feature_importances(df, clf, title, filename, bucket='s3://chorus-analytics-swamp/Pole-Retest-Analysis/cluster-analysis/', subfolder='pole-inspection/'):
    s3 = s3fs.S3FileSystem(anon=False)
    pathname = bucket + subfolder + filename
    
    importances = clf.feature_importances_
    #std = np.std([clf.feature_importances_], axis=0)
    indices = np.argsort(importances)[::-1]
    outline = title + "\n\n"
    maxlen = df[feature_list].columns[indices].str.len().max()
    outline += "Rank      Feature" + " "*(maxlen) + "Importance\n"
    outline += "-"*8 + "-"*(df[feature_list].columns[indices].str.len().max() + 19) + "\n"

    for f in range(df[feature_list].shape[1]):
        if df[feature_list].columns[indices[f]] == "donkey":
            break
        outline += str(str(f+1) + ' '*(10-len(str(f+1))) + str(df[feature_list].columns[indices[f]]) +  ' '*(27 - df[feature_list].columns[indices].str.len()[f]) + str(round(importances[indices[f]], 3))) + "\n"
    print(outline)

    with s3.open(pathname, 'w') as g:
        print(outline, file=g)
    

def generate_rules_outputs(df, clf, title, filename, feature_list, cluster_name, 
                           bucket='s3://chorus-analytics-swamp/Pole-Retest-Analysis/cluster-analysis/',
                           subfolder='pole-inspection/', max_depth=6, show_weights=False):
    s3 = s3fs.S3FileSystem(anon=False)
    pathname = bucket + subfolder + filename
    tree_rules = export_text(clf, feature_names=feature_list, max_depth=max_depth, show_weights=show_weights)
    print(title)
    print("-"*len(title) + "\n")
    print(tree_rules)    
    with s3.open(pathname, 'w') as g:
        print(title, file=g)
        print("-"*len(title) + "\n", file=g)
        print(tree_rules, file=g)

def create_tree_graph(clf, title,
                           feature_names,
                           class_names=['Binary Cluster 0','Other'],
                           bucket = "chorus-analytics-swamp",
                           s3_folder = "Pole-Retest-Analysis/cluster-analysis/pole-inspection",
                           max_depth=4,
                           filled=True,
                           leaves_parallel=True,
                           impurity=True, 
                           rounded=True, proportion=True,
                           showit=True):
    filename = title + ".png"

    key = s3_folder + "/" + filename

    dot_data = export_graphviz(clf,
                               max_depth=max_depth,
                               feature_names = feature_list,
                               class_names = class_names, 
                               filled=True,
                               leaves_parallel=True,
                               impurity=True, 
                               rounded=True, proportion=True)

    graph = graphviz.Source(dot_data, format='png')
    
    graph.render(filename=title, directory=None, view=False, 
           cleanup=False, format='png', renderer=None, 
           formatter=None, quiet=False, 
           quiet_view=False)
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).upload_file(filename, key)
    if showit:
        img = cv2.imread(filename)
        plt.figure(figsize = (20, 20))
        plt.imshow(img)
    
    os.remove(title)
    os.remove(filename)



# Add a normalized random dummy variable "donkey" in the set of predictors.
df["donkey"] = np.random.normal(loc=1,scale=1,size=len(df))

clf0 = get_classifier(df, feature_list, cluster_binary_target=df['Binary Cluster 0'], n_estimators = 100, max_depth=6)
clf1 = get_classifier(df, feature_list, cluster_binary_target=df['Binary Cluster 1'], n_estimators = 100, max_depth=6)
clf2 = get_classifier(df, feature_list, cluster_binary_target=df['Binary Cluster 2'], n_estimators = 100, max_depth=6)
clf3 = get_classifier(df, feature_list, cluster_binary_target=df['Binary Cluster 3'], n_estimators = 100, max_depth=6)



# Define list of features.

feature_list = [ 'pole_age',
                 'mt_softwood_old',
 'mt_softwood_new',
 'mt_hardwood',
	'gs_grass',
	'gs_asphalt',
	'gs_concrete',
	'gs_other',
	'gs_unknown',
	'foundation_integrity',
	'pole_vertical',
	'vegetation',
	'rd_lt10',
	'rd_1030',
	'rd_gt30',
	'rd_unknown',
	'rot_decay_portion',
	'impact_damage',
	'fire_damage',
	'saw_damage',
	'horizontal_cracking',
	'checking_length',
	'checking_width',
	'headsplit_length',
	'headsplits_width',
	'non_std_penetrations',
    'donkey']

# Create importance lists




generate_rules_outputs(df, clf0, "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 0", 
                       "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 0 Logic Flow.txt",
                       feature_list, cluster_name='Binary Cluster 0', max_depth=3, show_weights=True)

generate_rules_outputs(df, clf1, "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 1", 
                       "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 1 Logic Flow.txt",
                       feature_list, cluster_name='Binary Cluster 1', max_depth=3, show_weights=True)

generate_rules_outputs(df, clf2, "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 2", 
                       "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 2 Logic Flow.txt",
                       feature_list, cluster_name='Binary Cluster 2', max_depth=3, show_weights=True)

generate_rules_outputs(df, clf3, "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 3", 
                       "Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 3 Logic Flow.txt",
                       feature_list, cluster_name='Binary Cluster 3', max_depth=3, show_weights=True)
                       
                       
                       
                       create_tree_graph(clf=clf0, title="Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 0 Graph",
                           feature_names=feature_list,
                           class_names=['Binary Cluster 0','Other'],
                           bucket = "chorus-analytics-swamp",
                           s3_folder = "Pole-Retest-Analysis/cluster-analysis/pole-inspection",
                           max_depth=3,
                           filled=True,
                           leaves_parallel=True,
                           impurity=True, 
                           rounded=True, proportion=True)


create_tree_graph(clf=clf1, title="Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 1 Graph",
                           feature_names=feature_list,
                           class_names=['Binary Cluster 1','Other'],
                           bucket = "chorus-analytics-swamp",
                           s3_folder = "Pole-Retest-Analysis/cluster-analysis/pole-inspection",
                           max_depth=3,
                           filled=True,
                           leaves_parallel=True,
                           impurity=True, 
                           rounded=True, proportion=True)

create_tree_graph(clf=clf2, title="Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 2 Graph",
                           feature_names=feature_list,
                           class_names=['Binary Cluster 2','Other'],
                           bucket = "chorus-analytics-swamp",
                           s3_folder = "Pole-Retest-Analysis/cluster-analysis/pole-inspection",
                           max_depth=3,
                           filled=True,
                           leaves_parallel=True,
                           impurity=True, 
                           rounded=True, proportion=True)

create_tree_graph(clf=clf3, title="Pole Inspection Case 1-1 (Yellow 4 clusters) Cluster 3 Graph",
                           feature_names=feature_list,
                           class_names=['Binary Cluster 3','Other'],
                           bucket = "chorus-analytics-swamp",
                           s3_folder = "Pole-Retest-Analysis/cluster-analysis/pole-inspection",
                           max_depth=3,
                           filled=True,
                           leaves_parallel=True,
                           impurity=True, 
                           rounded=True, proportion=True)