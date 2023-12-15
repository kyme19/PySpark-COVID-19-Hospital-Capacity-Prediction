# PySpark COVID-19 Hospital Capacity Prediction
This repository contains PySpark code to extract COVID-19 hospital bed data from HDFS, build a model to predict hospital capacity needs, and evaluate the regression model.

## Ingest the data into Hadoop DFS Data lake 

hdfs dfs -copyFromLocal C:/Users/user/Downloads/kencovid.csv /kencovid.csv

in your case it might be 
hdfs dfs -copyFromLocal /path/to/kencovid.csv /kencovid
