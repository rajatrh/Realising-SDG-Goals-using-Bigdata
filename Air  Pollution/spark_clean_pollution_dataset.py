"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande
Description: 
Concepts Used:

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

from pyspark.sql import SparkSession
import sys
import ast
import csv
from itertools import islice

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

# Pollutant files to be processed
filePollutant = ast.literal_eval(sys.argv[1])

# Pollutant name
pollutant = sys.argv[2]

# Condition - 8 HOUR AVERAGE, 1 HOUR AVERAGE etc
condition = sys.argv[3]

rdd = sc.textFile(filePollutant).mapPartitionsWithIndex(
    lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

def append(x, county=True):
    if county:
        if len(x) == 1:
            x = '00' + x
        elif len(x) == 2:
            x = '0' + x
    else:
        if len(x) == 1:
            x = '0' + x
    return x

def check(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Keep the required columns only and clean the dataset
rdd.map(lambda x: (append(x[0], False), append(x[1]), x[2], x[3], x[9], x[11], x[19]))\
    .filter(lambda x: x[4] == condition and check(x[6]))\
        .map(lambda x: ((x[0]+x[1], x[5]), (float(x[6]), 1)))\
            .reduceByKey(lambda x, y: ((x[0] + y[0]), (x[1]+y[1])))\
                .map(lambda x: (x[0], (x[1][0]/x[1][1])))\
                    .map(lambda x: x[0]+ (x[1],)).map(lambda x: ','.join(str(s) for s in x))\
                        .saveAsTextFile('pollutant_processed' + pollutant)

