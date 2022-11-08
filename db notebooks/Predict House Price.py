# Databricks notebook source
storage_account_name = "pedro01b9ac"
storage_account_access_key = "g/hjC+VKhFWBaC7xUx/eMpOUif3MrrdK1m5LDlBjAg81yHJ3vphwRdLNlEavBWhyk0dyzRpE3BjU+ASt9dgksQ=="
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# import data
file_location = "wasbs://databricks-data@pedro01b9ac.blob.core.windows.net/housePrices_dataset_clean.csv/"
file_type = "csv"

df = spark.read.format(file_type).option("inferSchema", "true").option("header", "true").load(file_location)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

select_features = df.drop('SalePrice')

features_names = select_features.columns

vectorAssembler = VectorAssembler(inputCols=features_names , outputCol='features')

new_df = vectorAssembler.transform(df)

# COMMAND ----------

train, test = new_df.randomSplit([0.8,0.2])

# COMMAND ----------

import mlflow
import numpy as np

mlflow.autolog()

# COMMAND ----------

with mlflow.start_run(run_name='house-price-regression'):
    
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
    
    lr = LinearRegression(featuresCol='features', labelCol='SalePrice')
    rf_evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="r2")
    
    lrModel = lr.fit(train)
    
    lr_preds = lrModel.transform(test)
    
    r2 = rf_evaluator.evaluate(lr_preds)
    
    mlflow.log_metric("r2", r2)
    
    
    print('R-squared', rf_evaluator.evaluate(lr_preds))

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol='SalePrice', featuresCol='features')

rf_model = rf.fit(train)
rf_predictions = rf_model.transform(test)
print('R-squared', rf_evaluator.evaluate(rf_predictions))
