from pyspark.sql import SparkSession
import csv
import os
import sys
from itertools import islice

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

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

# Pollutant to be processed
filePollutant = sys.argv[1]
# Pollutant name
pollutant = sys.argv[2]

rdd = sc.textFile(filePollutant).mapPartitionsWithIndex(
    lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

# Keep the required columns only and clean the dataset
rdd.map(lambda x: (append(x[0], False), append(x[1]), x[2], x[3], x[9], x[11], x[19]))\
   .filter(lambda x: x[4] == '8-HR RUN AVG END HOUR')\
   .map(lambda x: ((x[0]+x[1], x[5]), (float(x[6]), 1)))\
   .reduceByKey(lambda x, y: ((x[0] + y[0]), (x[1]+y[1])))\
   .map(lambda x: (x[0], (x[1][0]/x[1][1])))\
   .map(lambda x: x[0]+ (x[1],))\
   .map(lambda x: ','.join(str(s) for s in x)).saveAsTextFile('/processed' + pollutant)
