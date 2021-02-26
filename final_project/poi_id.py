#!/usr/bin/python

import numpy as np
from numpy import asarray
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle
import sys
from warnings import simplefilter

sys.path.append("../tools/")

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

simplefilter(action='ignore') # ignore some sklearn future deprecation messages
test_algorithms = False
test_with_custom_features = True

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# List of all features for both exploration, and final classifiers
if(test_with_custom_features):
    features_list = ['poi','salary', 'exercised_stock_options',
                    'total_stock_value', 'expenses',
                    'exercised_bonus_mult', 'from_ratio_to_poi']
else:
    features_list = ['poi','salary', 'exercised_stock_options',
                 'total_stock_value', 'expenses']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
# Remove TOTAL item and the unknown name of TRAVEL AGENCY IN THE PARK
# Remove entry with all NaN values
to_remove = ['TOTAL', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']

for item in to_remove:
    data_dict.pop(item, 0)

def is_nan(value):
    """
    checks if feature value is 'NaN'
    returns true or false
    """
    is_nan = math.isnan(float(value))
    return is_nan

def pop_from_data(my_features, outlier_list):
    """
    remove outliers in outlier list
    return new dataset with outliers taken out
    """
    for person in outlier_list:
        try:
            my_features.pop(person)
        except:
            print('{0} not found to remove'.format(person))
    return my_features

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
data_array = [] #data to feed to classifiers
poi_array = [] # 0 or 1 to feed to classifier based on poi

def create_new_features(my_features):
    """
    creates new features 'salary_mult_bonus', 'exercised_bonus_mult', and 'from_ratio_to_poi'
    expects data dictionary
    returns dictionary dataset with new features within each person
    """
    salary_mult_bonus_feature = 'salary_mult_bonus'
    exercised_stock_sal_mult_feature = 'exercised_bonus_mult'
    message_ratio_feature = 'from_ratio_to_poi'
    list_to_pop = []
    for person in my_features:
        if(not is_nan(my_features[person]['total_stock_value'])):
            my_features[person]['total_stock_value'] = abs(my_features[person]['total_stock_value'])
        # checks if dependent features are present, builds new feature
        if(is_nan(my_features[person]['from_messages'])):
            my_features[person][message_ratio_feature] = 'NaN'
        else:
            sent_total = my_features[person]['from_messages']
            if(is_nan(my_features[person]['from_this_person_to_poi'])):
                my_features[person][message_ratio_feature] = 'NaN'
            else:
                sent_to_poi = my_features[person]['from_this_person_to_poi']
                from_ratio_to_poi = float(sent_to_poi) / float(sent_total)
                my_features[person][message_ratio_feature] = from_ratio_to_poi
        if(is_nan(my_features[person]['bonus'])):
            my_features[person]['salary_mult_bonus'] = 'NaN'
        else:
            bonus = my_features[person]['bonus']
            if(is_nan(my_features[person]['salary'])):
                my_features[person][salary_mult_bonus_feature] = 'NaN'
            else:
                salary = my_features[person]['salary']
                salary_mult_bonus = bonus * salary
                my_features[person][salary_mult_bonus_feature] = salary_mult_bonus
        if(is_nan(my_features[person][salary_mult_bonus_feature]) or is_nan(my_features[person]['exercised_stock_options'])):
            my_features[person][exercised_stock_sal_mult_feature] = 'NaN'
        else:
            exercised_and_bonus = my_features[person]['exercised_stock_options'] * my_features[person][salary_mult_bonus_feature]
            my_features[person][exercised_stock_sal_mult_feature] = exercised_and_bonus
        # builds data for test visualization
        if(my_features[person]['exercised_stock_options'] != 'NaN' and my_features[person][salary_mult_bonus_feature] != 'NaN'):
            feature_data = []
            feature_data.append(my_features[person]['exercised_stock_options'])
            feature_data.append(my_features[person][salary_mult_bonus_feature])
            data_array.append(feature_data)
            # separates the persons on interest on the plot (Blue is POI)
            if(my_features[person]['poi']):
                poi_array.append(1)
            else:
                poi_array.append(0)
        for feature in features_list:
            if(is_nan(my_features[person][feature])):
                if person not in list_to_pop:
                    list_to_pop.append(person)

    # remove outliers from the dataset
    my_features = pop_from_data(my_features, list_to_pop)

    return my_features

# data with newly created features
my_features = create_new_features(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_features, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

clf = DecisionTreeClassifier(min_samples_split=20, criterion='entropy')

def get_classifier_list():
    """
    creates list of classifers and returns the classifier list:
    Random Forest
    Linear SVC
    Decision Tree
    """
    print('### Settings Classifiers ###')
    classifier_list = []

    params_rand_tree = {"n_estimators":[5, 7], "criterion": ('gini', 'entropy')}
    classifier_list.append((RandomForestClassifier(), params_rand_tree))
    params_linearsvm = {"C": [0.8, 1, 5, 10, 100], "tol": [10**-1, 10**-10]}
    classifier_list.append((LinearSVC(), params_linearsvm))
    params_tree = {"min_samples_split":[2, 5, 10, 20], "criterion": ('gini', 'entropy')}
    classifier_list.append((DecisionTreeClassifier(), params_tree))
    params_linearsvm = {"C": [0.5, 1, 5, 10], "tol":[10**-1, 10**-10]}
    classifier_list.append( (LinearSVC(), params_linearsvm) )

    return classifier_list

def find_best_params(classifier_list, features_train, labels_train):
    """
    tunes for the best parameters from the range given.
    scores the results, appends to list and returns the list
    """
    print('### Finding Best Parameters for Algorithms ###')
    best_params = []
    # check all classifiers, score them based on recall score, used gridsearch to pick the top score for params
    for clf, params in classifier_list:
        scorer = make_scorer(recall_score)
        classifier = GridSearchCV(clf, params, scoring=scorer)
        classifier = classifier.fit(features_train, labels_train)
        classifier = classifier.best_estimator_
        best_params.append(classifier)

    return best_params

def final_eval(features, labels, labels_train, labels_test, clf):
    """
    evaluate the algorithms with ranged parameters
    iterates through checking the recall and precision scores with previous scores to pick the best one
    since it goes off random splits, it can be unpredictable, if no match is found,
    it goes with the most common outcome as a default
    """
    print('### Starting Final Evaluation ###')
    recall_scores = []
    precision_scores = []
    classifier_list = get_classifier_list()

    final_classifiers = find_best_params(classifier_list, features_train, labels_train)

    # score each algorithm and check for the best based on recall score
    for classifier in final_classifiers:
        prediction = classifier.predict(features_test)
        recall = recall_score(labels_test, prediction)
        recall_scores.append(recall)
        precision = precision_score(labels_test, prediction)
        precision_scores.append(precision)
        if((precision >= max(precision_scores) and precision > 0.4) and recall >= max(recall_scores) and recall > 0.4):
            clf = classifier
            print(clf)
            dump_classifier_and_data(clf, my_dataset, features_list)
        else:
            dump_classifier_and_data(clf, my_dataset, features_list)
    print("### Completed, you can now run tester.py ###")

# data for data exploration visualizations
X = data_array
y = poi_array

def create_classifiers():
    """
    creates and returns list of classifers only used for data exploration visualization
    for use in test plots only
    """
    classifier_list = [KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=3, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=4, n_estimators=10, max_features=2),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]
    return classifier_list

def create_classifier_name_list():
    """
    creates and returns list of classifier names
    for use in test plots only
    """
    classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
        "Random Forest", "AdaBoost", "Naive Bayes", "Lin Dscrm Anlys",
        "Quadrtc Dscrm Anlys"]
    return classifier_names

def process_data(my_features, X, y):
    """
    visualization adapted from sklearn example documentation
    Process and return the dataset
    iterates over the dataset, splits the data into training data and testing data
    for use in test plots only
    """
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]
    
    figure = plt.figure(figsize=(27, 9))
    i = 1
    h = 0.2
    classifiers = create_classifiers()
    names = create_classifier_name_list()
    for data in datasets:
        X, y = data
        X = StandardScaler().fit_transform(X) 
        X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 4)
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ax.scatter(X_training[:, 0], X_training[:, 1], c=y_training, cmap=cm_bright)
        ax.scatter(X_testing[:, 0], X_testing[:, 1], c=y_testing, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_training, y_training)
            score = clf.score(X_testing, y_testing)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_training[:, 0], X_training[:, 1], c=y_training, cmap=cm_bright)
            # and testing points
            ax.scatter(X_testing[:, 0], X_testing[:, 1], c=y_testing, cmap=cm_bright,
                        alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1
    figure.subplots_adjust(left=.02, right=.98)
    plt.show()

# checks if testing visualization, past exploration should be false
if (test_algorithms):
    process_data(my_features, X, y)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# tune, evaluate, and dump the data
final_eval(features, labels, labels_train, labels_test, clf)
