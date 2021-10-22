#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    tuple_list = []
    for i in range(len(ages)):
        a = ages[i][0]
        n = net_worths[i][0]
        p = predictions[i][0]
        e = (p -n)**2
        tuple_list.append((a,n,e)) 

    cleaned_data = sorted(tuple_list, key = lambda x: x[2], reverse = True)
    limit = int(len(net_worths)*0.1)
    return cleaned_data[limit:]


