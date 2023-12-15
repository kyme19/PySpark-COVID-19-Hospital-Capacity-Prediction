# PySpark COVID-19 Hospital Capacity Prediction
This repository contains PySpark code to extract COVID-19 hospital bed data from HDFS, build a model to predict hospital capacity needs, and evaluate the regression model.

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
### Step Three: Reading the CSV file from the HDFS into a DataFrame

```python
df = spark.read.option("header",True).csv("hdfs://localhost:9000/kencovid.csv")
```
