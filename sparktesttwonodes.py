from pyspark.sql import SparkSession
import numpy as np

# Test Spark connection
spark = SparkSession.builder \
    .appName("ParticleSimulationTest") \
    .master("spark://130.229.147.250:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

print("Spark session created successfully!")
print(f"Spark version: {spark.version}")
print(f"Number of executors: {len(spark.sparkContext.statusTracker().getExecutorInfos())}")

# Test a simple RDD operation
test_data = [1, 2, 3, 4, 5]
rdd = spark.sparkContext.parallelize(test_data)
result = rdd.map(lambda x: x * 2).collect()
print(f"Test RDD result: {result}")

spark.stop()
print("Spark test completed!")