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


