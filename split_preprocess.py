#spliting data into train and test

from pyspark.sql.functions import rand
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Vector Assemble Features 
# trying to train a linear regression model to predict the number of confirmed cases

from pyspark.ml.feature import VectorAssembler 
assembler = VectorAssembler(
    inputCols=["Regular Isolation Beds Available", "Recommended ICU/Critical Care beds for Isolation"],
    outputCol="features")  
train_vect = assembler.transform(train_df) 
test_vect = assembler.transform(test_df)
