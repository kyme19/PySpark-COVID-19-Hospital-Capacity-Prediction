# train model
from pyspark.ml.regression import LinearRegression
label_col = "Recommended ICU/Critical Care beds for Isolation" 
lr = LinearRegression(featuresCol="features", labelCol=label_col)
fitted_model = lr.fit(train_vect)


#Evaluate model

test_results = fitted_model.evaluate(test_vect)
print(test_results.rootMeanSquaredError)
print(test_results.r2)


#visualize model

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
