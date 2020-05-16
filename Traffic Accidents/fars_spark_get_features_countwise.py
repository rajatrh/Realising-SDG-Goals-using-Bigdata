"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande
Description: 
Concepts Used:

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

import os
from pyspark import SparkContext
import csv
import sys
import datetime
import numpy as np
from itertools import islice
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

sc = SparkContext('local[*]', 'pyspark tutorial')

personsMerged = sys.argv[1]
accidentFile = sys.argv[2]

def formNewCols(x):
  print(x[1])
  personSplit = [0,0,0,0,0,0,0,0]

  for p in x[1]:
    # Grouping by age
    age = int(p[0])
    if age not in [99, 999]:
      if age < 16:
        personSplit[0] +=1
      elif age < 30:
        personSplit[1] +=1
      elif age < 60:
        personSplit[2] +=1
      else:
        personSplit[3] +=1

    # Grouping by gender
    sex = int(p[1])
    if sex == 1:
      personSplit[4] +=1
    elif sex == 2:
      personSplit[5] +=1

    # Grouping by restraint used
    ru = int(p[2])
    # Not used
    if ru in [17,20] or ((int(x[0][-4:]) < 2009) & ru == 0):
      personSplit[6] +=1
    # Used Restraint
    elif ru < 17:
      personSplit[7] +=1
      
  return (x[0],personSplit)

def evaluate(x):
  res = []
  for i,col in enumerate(x):
    try:
      if i in [20]:
        res.append(col)
      else:
        res.append(eval(col))
    except:
      res.append(col)
  return res

def featureExtractionCountyWise(countyData, ignore_col_set = {9,16,20,44,45,31}, yCol=45):
  county = countyData[0]
  try:
    df_vals = countyData[1]
    df_vals = np.asarray(df_vals, dtype = object)
    feature_set = []
    column_val_list = np.transpose(df_vals)
    
    x_trans = []
    idx_dict = {}
    req_col_length = len(column_val_list[0]) * 0.1

    idx_count = 0
    for i, col_vals in enumerate(column_val_list):
      if i in ignore_col_set:
        pass
      else:
        val_count = Counter(col_vals)
        count = 0
        for value in val_count:
          if val_count[value] >= req_col_length:
            count += 1;
          if count > 1:
            x_trans.append(list(col_vals))
            idx_dict[idx_count] = i
            idx_count += 1
            break
    
    if (x_trans) and len(x_trans[0]) < 10:
      return (county, ([], {}, {}, []))

    x_trans = np.asarray(x_trans, dtype = object)
    
    x = np.transpose(x_trans)
    y = np.array(list(map(lambda el:[el], column_val_list[yCol])))
    
    knn = KNeighborsClassifier(n_neighbors=1)

    sfs1 = SFS(knn, 
              k_features=10, 
              forward=True,
              floating=True, 
              verbose=2,
              scoring='accuracy',
              cv=5)

    sfs1 = sfs1.fit(x, y)

    cols_idx_set = set(sfs1.subsets_[10]['feature_idx'])

    
    for i, cols in enumerate(x_trans):
      if i in cols_idx_set:
        feature_set.append((i, Counter(cols)))
    
    idx_list = []
    for cols in cols_idx_set:
      idx_list.append(idx_dict.get(cols))

    return (county, (feature_set, cols_idx_set, idx_dict, idx_list))
  except:
    return (county, ([], {}, {}, []))

personRDD = sc.textFile(personsMerged).mapPartitionsWithIndex(
  lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

aggPersonRDD = personRDD.map(lambda x: ((x[75]), (x[5], x[6], x[9]))).groupByKey()\
.mapValues(list).map(lambda x: formNewCols(x))


aRDD = sc.textFile(accidentFile).mapPartitionsWithIndex(
lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

## Set aside the id (so as to merge)
accidentRDD = aRDD.map(lambda x: (x[45],evaluate(x[1:len(x)])))

merged_accidentsRDD = accidentRDD.join(aggPersonRDD).map(lambda x:  x[1][0] + x[1][1])

merged_accidentsRDD.saveAsTextFile('accidentsFinal')

## Group over location (Location - col 21)
groupbylocRDD = merged_accidentsRDD.map(lambda x: (x[20], x)).groupByKey().mapValues(list)

from collections import Counter

groupbylocRDD.map(lambda x: featureExtractionCountyWise(x)).saveAsTextFile('countyFeatures')

