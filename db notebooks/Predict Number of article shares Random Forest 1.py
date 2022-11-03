# Databricks notebook source
storage_account_name = "pedro01b9ac"
storage_account_access_key = "g/hjC+VKhFWBaC7xUx/eMpOUif3MrrdK1m5LDlBjAg81yHJ3vphwRdLNlEavBWhyk0dyzRpE3BjU+ASt9dgksQ=="
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# import data
file_location = "wasbs://databricks-data@pedro01b9ac.blob.core.windows.net/OnlineNewsPopularity.csv/"
file_type = "csv"

df = spark.read.format(file_type).option("inferSchema", "true").option("header", "true").load(file_location)

# COMMAND ----------

for column in df.columns:
    df = df.withColumnRenamed(column, column.replace(' ',''))

# COMMAND ----------

df = df.drop('url')

# COMMAND ----------

select_features = df.drop('shares')

features_names = select_features.columns

# COMMAND ----------

type(features_names)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=features_names , outputCol='features')

new_df = vectorAssembler.transform(df)

# COMMAND ----------

train, test = new_df.randomSplit([0.8,0.2])

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

rf = RandomForestRegressor(labelCol='shares', featuresCol='features', maxDepth=2, maxBins=5, numTrees=5)
rf_evaluator = RegressionEvaluator(labelCol="shares", predictionCol="prediction", metricName="r2")

rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)
print('R-squared', rf_evaluator.evaluate(rf_predictions))

# COMMAND ----------

import matplotlib.pyplot as plt

r2 = rf_evaluator.evaluate(rf_predictions)

rf_result = rf_predictions.toPandas()

plt.plot(rf_result.shares, rf_result.prediction, 'bo')
plt.xlabel('Shares')
plt.ylabel('Prediction')
plt.suptitle("Model Performance R-Square: %f" % r2)
plt.show()

# COMMAND ----------

rf_evaluator.evaluate(rf_predictions, {rf_evaluator.metricName:"mae"})

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

predictions = np.array(rf_predictions.select('prediction').collect())
y_test = np.array(test.select('shares').collect())


fig = plt.figure(figsize= (20,10))
plot_size = 100

plt.plot(predictions[:plot_size], label = "Predicted", color='orange', linestyle='dashed') 
plt.plot(y_test[:plot_size], label= "Original", color='blue') 
