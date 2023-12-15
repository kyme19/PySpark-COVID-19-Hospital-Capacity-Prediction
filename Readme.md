# PySpark COVID-19 Hospital Capacity Prediction
This repository contains PySpark code to extract COVID-19 hospital bed data from HDFS, build a model to predict hospital capacity needs, and evaluate the regression model.

# Phase 1
## Step One: Ingest the data into Hadoop DFS Data lake 

```bash

hdfs dfs -copyFromLocal C:/Users/user/Downloads/kencovid.csv /kencovid.csv

in your case it might be 

hdfs dfs -copyFromLocal /path/to/kencovid.csv /kencovid
```

imgs/image one.png
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






# Phase 2: Pre-process the extracted data 
## The following are techniques we used to pre-process the extracted CSV data in the pyspark DataFrame

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
### image nine


# phase 3: Applying predective analytics 
## Step One: pre-process data: Import libraries and split data into training and test sets

```python
from pyspark.sql.functions import when  
from pyspark.sql.types import StringType, IntegerType

df = df.withColumn("Recommended ICU/Critical Care beds for Isolation",  
                   df["Recommended ICU/Critical Care beds for Isolation"].cast(StringType()))

df = df.withColumn("Recommended ICU/Critical Care beds for Isolation",
                   when(df["Recommended ICU/Critical Care beds for Isolation"].rlike("^[0-9]+$"),
                        df["Recommended ICU/Critical Care beds for Isolation"]
                       ).otherwise(None))
                       
df = df.withColumn("Recommended ICU/Critical Care beds for Isolation",
                   df["Recommended ICU/Critical Care beds for Isolation"].cast(IntegerType()))

```

## Step two: Split Data

```python
from pyspark.sql.functions import rand
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
```
### image ten

## Step three: Vector assemble features 
Here were trying to train a linear regression model to predict confirmed cases:

```python
from pyspark.ml.feature import VectorAssembler 
assembler = VectorAssembler(
    inputCols=["Regular Isolation Beds Available", "Recommended ICU/Critical Care beds for Isolation"],
    outputCol="features")  
train_vect = assembler.transform(train_df) 
test_vect = assembler.transform(test_df)
```

### image eleven 
## Step 4: Train model

```python
from pyspark.ml.regression import LinearRegression
label_col = "Recommended ICU/Critical Care beds for Isolation" 
lr = LinearRegression(featuresCol="features", labelCol=label_col)
fitted_model = lr.fit(train_vect)
```

### image 12 
A close up view on evaluated data 
### image 13


## Step 5: Evaluate model

```python
test_results = fitted_model.evaluate(test_vect)
print(test_results.rootMeanSquaredError)
print(test_results.r2)
```
### image 14

## Step 6: Visualize the model

Here are the key steps to visualize and test the linear regression model we built, we used the fitted model to make predictions on the test data

```python
test_pred = fitted_model.transform(test_vect)
true_vals = test_pred.select("Recommended ICU Beds").collect()
pred_vals = test_pred.select("prediction").collect()

import matplotlib.pyplot as plt
true_x = [r.Recommended_ICU_Beds for r in true_vals]  
pred_x = [r.prediction for r in pred_vals]
plt.scatter(true_x, pred_x)
plt.plot([0, 50], [0, 50], c='red') 
plt.title("True vs Predicted")
plt.show()
```
image 15


### step 7: Test the model 
```python
test_df = spark.createDataFrame([(500, 100, 50)], ["Regular Isolation Beds Available", "Total ICU Beds", "Ventilators Available"])
model.transform(test_df).select("prediction").show()
```
