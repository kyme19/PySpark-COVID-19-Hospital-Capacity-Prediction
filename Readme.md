# PySpark COVID-19 Hospital Capacity Prediction
This repository contains PySpark code to extract COVID-19 hospital bed data from HDFS, build a model to predict hospital capacity needs, and evaluate the regression model.

# Phase 1
## Step One: Ingest the data into Hadoop DFS Data lake 

```bash

hdfs dfs -copyFromLocal C:/Users/user/Downloads/kencovid.csv /kencovid.csv

in your case it might be 

hdfs dfs -copyFromLocal /path/to/kencovid.csv /kencovid
```

### image one 
#### in hadoop
### image two

## Step Two: Use pyspark package to extract data from the data lake 
#### first we begin by starting a SparkSession 

```python
import pyspark 
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("csv_extraction").getOrCreate()
```
### image three  

### Step Three: Reading the CSV file from the HDFS into a DataFrame

the option tell spark to treat the first row as header 

```python
df = spark.read.option("header",True).csv("hdfs://localhost:9000/kencovid.csv")
```
### image four  

Now the DataFrame df contains the contents of the CSV file. You can perform analysis and transformations on it. For example, to print the schema:
```python
df.printSchema()
```
### image five 

to print the first few rows 

```python
df.show(5)
```
### image six






# Phase 2
## Choosing the appropriate techniques to Pre- process the extracted data the following are techniques we used to pre-process the extracted CSV data in the pyspark DataFrame

##Step One
### first we start by handling the missing values

```python
from pyspark.sql.functions import when, count, col
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
df = df.fillna({'Regular Isolation Beds Available': 0}) 
```
### image seven

##Step two
### Convert columns to appropriate data types

```python
from pyspark.sql.types import IntegerType, StringType
df = df.withColumn("Regular Isolation Beds Available", df["Regular Isolation Beds Available"].cast(IntegerType()))
df = df.withColumn("County", df["County"].cast(StringType())) 
```
### image eight

## Step three
### filter outlier values / incorrect values 

```python
from pyspark.sql.functions import col

min_val = df.agg({"Regular Isolation Beds Available": "min"}).first()[0]
max_val = df.agg({"Regular Isolation Beds Available": "max"}).first()[0]

df_cleaned = df.filter(col("Regular Isolation Beds Available") > min_val) \
             .filter(col("Regular Isolation Beds Available") < max_val)

```

To display the DataFrame and print the first few rows after applying the transformations, we used 

```python
df.printSchema() # print schema
df.show(5) # print first 5 rows
```
