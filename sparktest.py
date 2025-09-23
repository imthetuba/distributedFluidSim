from pyspark.sql import SparkSession, Row
from pyspark import SparkContext, SparkConf

#$ ./bin/pyspark --master "local[4]" --py-files test.py
kmvkdk
"""
two types of operations:
transformations
    - create new dataset from exisiting one
    - i.e. map 
actions
    - return value to driver program after computation on dataset
    - i.e. reduce

Spark is LAZY - Transformations are only computed after action has been called

Persist (cache) - keep elements on cluster for faster access

"""

conf = SparkConf().setAppName("Test 1").setMaster(master)
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()


df = spark.createDataFrame([
    Row(Name="John", Salary=150, DOB='990811'),
    Row(Name="Erik", Salary=200, DOB='981012'),
    Row(Name="Karl", Salary=500, DOB='000312')
])

data = [1, 2, 3, 4, 5]
distData = sc.parallelize(data)


