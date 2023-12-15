#handling missing values

from pyspark.sql.functions import when, count, col
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
df = df.fillna({'Regular Isolation Beds Available': 0}) 

#convert columns to appropriate data types

from pyspark.sql.types import IntegerType, StringType
df = df.withColumn("Regular Isolation Beds Available", df["Regular Isolation Beds Available"].cast(IntegerType()))
df = df.withColumn("County", df["County"].cast(StringType())) 

#filter outlier or incorrect values
from pyspark.sql.functions import col

min_val = df.agg({"Regular Isolation Beds Available": "min"}).first()[0]
max_val = df.agg({"Regular Isolation Beds Available": "max"}).first()[0]

df_cleaned = df.filter(col("Regular Isolation Beds Available") > min_val) \
             .filter(col("Regular Isolation Beds Available") < max_val)


#show the dataframe 
df.printSchema() # print schema

df.show(5) # print first 5 rows

