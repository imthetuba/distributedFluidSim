from pyspark.sql import SparkSession, Row



#test
# spark-submit --master local[3] sparktest.py

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

spark = SparkSession.builder.appName("Spark example").getOrCreate()

data = [("John", 150,'990811'),("Erik", 200,'981012'), ("Karl", 500, '000312') ]
df = spark.createDataFrame(data, ["Name", "Salary", "DOB"])

df.show()

spark.stop()


