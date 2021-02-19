#!/usr/bin/python

import numpy as np
import math
import matplotlib.pyplot
import pickle
import sys

sys.path.append("../tools/")

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# List of all features for both exploration, and final classifiers
features_list = ['poi','salary', 'total_payments', 'exercised_stock_options',
                 'bonus', 'restricted_stock', 'total_stock_value', 'expenses',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# Remove TOTAL item and the unknown name of TRAVEL AGENCY IN THE PARK
# Remove entry with all NaN values
to_remove = ['TOTAL', 'LOCKHART EUGENE E', 'THE TRAVEL AGENCY IN THE PARK']

for item in to_remove:
    data_dict.pop(item, 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_features = data_dict

salary_mult_bonus_feature = 'salary_mult_bonus'
message_ratio_feature = 'from_ratio_to_poi'

has_message_feature = []
has_bonus_feature = []

# Get percentage of messages sent to a POI out of the total messages sent
for person in my_features:
    if(math.isnan(float(my_features[person]['from_messages']))):
        my_features[person][message_ratio_feature] = 'NaN'
    else:
        sent_total = my_features[person]['from_messages']
        if(math.isnan(float(my_features[person]['from_this_person_to_poi']))):
            my_features[person][message_ratio_feature] = 'NaN'
        else:
            has_message_feature.append(person)
            sent_to_poi = my_features[person]['from_this_person_to_poi']
            from_ratio_to_poi = float(sent_to_poi) / float(sent_total)
            my_features[person][message_ratio_feature] = from_ratio_to_poi
    if(math.isnan(float(my_features[person]['bonus']))):
        my_features[person]['salary_mult_bonus'] = 'NaN'
    else:
        bonus = my_features[person]['bonus']
        if(math.isnan(float(my_features[person]['salary']))):
            my_features[person][salary_mult_bonus_feature] = 'NaN'
        else:
            has_bonus_feature.append(person)
            salary = my_features[person]['salary']
            salary_mult_bonus = bonus * salary
            my_features[person][salary_mult_bonus_feature] = salary_mult_bonus

features_list.append(message_ratio_feature)
features_list.append(salary_mult_bonus_feature)

print('Has Message Feature: {0}'.format(len(has_message_feature)))
print('Has Bonus Feature: {0}'.format(len(has_bonus_feature)))

### Extract features and labels from dataset for local testing
data = featureFormat(my_features, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# TODO: Add classifiers
clf = GaussianNB()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)
