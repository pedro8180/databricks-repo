# Databricks notebook source
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# COMMAND ----------

# get data
storage_account_name = "pedro01b9ac"
storage_account_access_key = "g/hjC+VKhFWBaC7xUx/eMpOUif3MrrdK1m5LDlBjAg81yHJ3vphwRdLNlEavBWhyk0dyzRpE3BjU+ASt9dgksQ=="
spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# import data
file_location = "wasbs://databricks-data@pedro01b9ac.blob.core.windows.net/OnlineNewsPopularity.csv/"
file_type = "csv"

df = spark.read.format(file_type).option("inferSchema", "true").option("header", "true").load(file_location)

for column in df.columns:
    df = df.withColumnRenamed(column, column.replace(' ',''))

df = df.drop('url')

x_train = df.drop('shares')
y_train = df['shares']

# COMMAND ----------

import numpy as np

x_train = np.array(x_train.collect())

# COMMAND ----------

y_train = np.array(df.select('shares').collect())

# COMMAND ----------

def train_keras_model(x,y):
    model = Sequential()
    model.add(Dense(100, input_shape=(x_train.shape[-1],), activation="relu", name="hidden_layer"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=.2)
    return model

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow

with mlflow.start_run():
    mlflow.tensorflow.autolog()
    
    
    train_keras_model(x_train, y_train)
    run_id = mlflow.active_run().info.run_id

# COMMAND ----------

run_id

# COMMAND ----------

logged_model = 'runs:/864c918a3737425787e2e558b0137550/model'

# COMMAND ----------

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

# COMMAND ----------

from pyspark.sql.functions import struct, col

new_df = df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))

# COMMAND ----------

new_df.display()

# COMMAND ----------

new_df.select("predictions").limit(200)

# COMMAND ----------

def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.div(residual, total))
    return r2

# COMMAND ----------

new_df.select("predictions").collect()

# COMMAND ----------

new_df.select("shares").count()

# COMMAND ----------

y_pred = np.array(new_df.select("predictions").collect())



# COMMAND ----------

y_pred = np.array(new_df.select("predictions").limit(3500))



y = np.array(new_df.select("shares").limit(3500).collect())

R_squared(y, y_pred)
