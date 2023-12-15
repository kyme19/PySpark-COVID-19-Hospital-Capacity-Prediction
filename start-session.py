import pyspark 
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("csv_extraction").getOrCreate()