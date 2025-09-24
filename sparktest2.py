from pyspark import SparkContext

sc = SparkContext.getOrCreate()

rdd = sc.parallelize(range(1, 1000000))
print(rdd.filter(lambda x: x % 2 == 0).count())

sc.stop()
