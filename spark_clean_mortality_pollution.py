from pyspark.sql import SparkSession
import sys
import csv
from itertools import islice


# Pollutant files to be processed
fileMortality = sys.argv[1]
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

rdd = sc.textFile(fileMortality)\
    .mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it).mapPartitions(lambda x: csv.reader(x))

# Modify the FIPS code
def append(x):
  x = str(int(float(x)))
  if len(x) == 4:
    x = '0' + x
  return x


rdd.filter(lambda x: x[4] != "United States")\
  .map(lambda x: (append(x[5]), x[6], x[8], x[12], x[14]))\
    .filter(lambda x: len(x[0]) > 2 )\
      .saveAsTextFile('processed_mortality')