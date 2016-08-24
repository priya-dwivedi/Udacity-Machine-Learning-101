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


import os
os.chdir(r"C:\Users\s6324900\Documents\ud120projects\tools")

import pickle

enron_data = pickle.load(open(r"C:\Users\s6324900\Documents\ud120projects\final_project\final_project_dataset.pkl", "r"))

print  enron_data["SKILLING JEFFREY K"]["bonus"]

# No of items 
print len(enron_data) 
print enron_data.items()

print len(enron_data.keys())

keys = enron_data.keys()
print (keys)

# Nof of Features
features_dict = enron_data["SKILLING JEFFREY K"] 
print len(features_dict)
#
# count = 0
# if enron_data[]["poi"] == 1:
#     count = count+1

# print count


print enron_data["PRENTICE JAMES"]["total_stock_value"]

print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]

count = 0
for person in enron_data:
    if enron_data[person]["poi"] == 1:
        count = count+1

print count

# No of quantified salaries
count = 0
for person in enron_data:
    if enron_data[person]["salary"] != "NaN":
        count = count+1

print count

# No of email address
count = 0
for person in enron_data:
    if enron_data[person]["email_address"] != "NaN":
        count = count+1

print count

# No of missing total_payments
count = 0
for person in enron_data:
    if enron_data[person]["total_payments"] == "NaN":
        count = count+1

#  POI with missing total_payments
count = 0
for person in enron_data:
    if enron_data[person]["poi"] == 1:
        if enron_data[person]["total_payments"] == "NaN":
            count = count+1

print count

#  Identify outlier on salary
sal =0
for person in enron_data:
    if enron_data[person]["salary"] > sal:
        sal = enron_data[person]["salary"]
    else:
        sal = sal

print sal

