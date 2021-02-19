#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np
import sys

sys.path.insert(0, '/home/scottyh/workspace/ud120-projects/final_project')
sys.path.insert(0, '/home/scottyh/workspace/ud120-projects/tools')

import poi_email_addresses
import feature_format 

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(len(poi_email_addresses.poiEmails()))

keys = []
enron_employees = []
enron_poi = []
missing_features = 0
missing_values = {} #feature: employee name
data = []

search = ['salary', 'bonus', 'total_stock_value']

# get employee list
for person in enron_data:
    enron_employees.append(person)
    individualData = enron_data[person]
    individualData['name'] = person
    data.append(individualData)
    if enron_data[person]['poi'] is True:
        enron_poi.append(person)

# get possible features
for employee in enron_employees:
    individual = enron_data[employee]
    for key in individual:
        if key not in keys:
            if key is not 'name':
                keys.append(key)
        try:
            individual[key]
            if individual[key] in [None, 'NaN', '']:
                if missing_values.has_key(key):
                    missing_values[key].append(employee)
                else:
                    missing_values[key] = [employee]
        except:
            missing_features + 1

for key in missing_values:
    print('{0}: {1}'.format(key, len(missing_values[key])))

# options(dictionary, features, remove_NaN=True, remove_all_zeroes=True, sort_keys=False)
ff = feature_format.featureFormat(enron_data, search)

#print(feature_format.targetFeatureSplit(ff))

# targetFeatureSplit(data)

print('Total Enron Employees In Dataset: {0}'.format(len(enron_employees)))
print('Total Possible Features Per Employee: {0}'.format(len(keys)))
print('Total Missing Values: {0}'.format(len(missing_values)))
print('Total Enron POIs: {0}'.format(len(enron_poi)))
print('Total POIs: {0}'.format(len(poi_email_addresses.poiEmails())))
#print('Missing values: {0}'.format(missing_values))

print(keys)

#print(missing_values)

print(enron_data['PRENTICE JAMES']['total_stock_value'])
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])
