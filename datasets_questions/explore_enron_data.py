#!/usr/bin/python3

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

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

for key, value in enron_data.items():
    #print value
    print(key, [item for item in value if item])
    break

count = 0  
for value in enron_data.values():
    if value['poi'] == 1:
        count += 1
print('there are {} POIs in the dataset'.format(count))


print(enron_data['PRENTICE JAMES']['total_stock_value'])