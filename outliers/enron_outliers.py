#!/usr/bin/python3

import joblib
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# names = []
# for i in data_dict.keys():
#     # be careful with null values; zeros haven't been removed from the data_dict
#     if float(data_dict[i]['bonus']) > 5000000 and data_dict[i]['bonus'] != 'NaN' \
#     and float(data_dict[i]['salary']) > 1000000 and data_dict[i]['salary'] != 'NaN':
#         names.append(i)
# print(names)



### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



