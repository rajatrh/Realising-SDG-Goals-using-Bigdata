"""
Memebers : Abhiram Kaushik, Ajay Gopal Krishna, Rajesh Prabhakar, Rajat R Hande
Description: 
Concepts Used:

Config: DataProc on GCP, Image: 1.4.27-debian9
        M(1): e2-standard-2 32GB
        W(3): e2-standard-4 64GB
"""

from pyspark.sql import SparkSession
import glob
import pandas as pd
import sys

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

# Result of Linear Regression merged into a single file.
regressionResultFiles = sys.argv[1]

overallRDD = sc.textFile(regressionResultFiles)

def parseIntoCSVrows(x):
  output = []
  county = int(x[0])
  pollutants = x[1]
  for p in pollutants:
    pollutant = p[0]
    years = p[1]
    aqi = p[2]
    dr = p[3]
    beta = p[4]
    for y in range(len(years)):
      output.append((county, pollutant, years[y], aqi[y], dr[y], beta))
  return output


overallRDD.map(lambda x: eval(x)).groupByKey().mapValues(list)\
.flatMap(parseIntoCSVrows).saveAsTextFile('/overallResult')