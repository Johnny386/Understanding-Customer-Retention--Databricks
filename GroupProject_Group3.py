# Databricks notebook source
# DBTITLE 1,Group Information
#Group Number: 3
#Group Members: Johnny CHREIM, Yaqing HU
#Academic Year: 2025S2
#Course: Big Data Tools

# COMMAND ----------

#Load functions
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import GBTClassificationModel, RandomForestClassificationModel, LogisticRegressionModel
import pandas as pd
import numpy as np
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Binarizer
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql import Row

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading Data

# COMMAND ----------

# DBTITLE 1,Reading Data
#Read in data: orders, products, order_items, order_payments, order_reviews, test_order_items, test_order_payments, test_orders, test_products

filePath1 = "dbfs:/FileStore/Big Data Project/orders.csv"
filePath2 = "dbfs:/FileStore/Big Data Project/products.csv"
filePath3 = "dbfs:/FileStore/Big Data Project/order_items.csv"
filePath4 = "dbfs:/FileStore/Big Data Project/order_payments.csv"
filePath5 = "dbfs:/FileStore/Big Data Project/order_reviews.csv"
filePath6 = "dbfs:/FileStore/Big Data Project/test_order_items.csv"
filePath7 = "dbfs:/FileStore/Big Data Project/test_order_payments.csv"
filePath8 = "dbfs:/FileStore/Big Data Project/test_orders.csv"
filePath9 = "dbfs:/FileStore/Big Data Project/test_products.csv"

orders=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath1)

products=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath2)

order_items=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath3)

order_payments=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath4)

order_reviews=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath5)

test_order_items=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath6)

test_order_payments=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath7)

test_orders=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath8)

test_products=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath9)

# COMMAND ----------

orders.createOrReplaceTempView("orders")
products.createOrReplaceTempView("products")
order_items.createOrReplaceTempView("order_items")
order_payments.createOrReplaceTempView("order_payments")
order_reviews.createOrReplaceTempView("order_reviews")
test_order_items.createOrReplaceTempView("test_order_items")
test_order_payments.createOrReplaceTempView("test_order_payments")
test_orders.createOrReplaceTempView("test_orders")
test_products.createOrReplaceTempView("test_products")

# COMMAND ----------

order_reviews.count()

# COMMAND ----------

distinct = order_reviews.select(count_distinct('order_id'), count_distinct('review_id')).show()

# COMMAND ----------

latest_reviews = order_reviews.groupBy("order_id").agg(max("review_answer_timestamp").alias("latest_timestamp"))
order_reviews = order_reviews.join(latest_reviews, (order_reviews["order_id"]==latest_reviews["order_id"])&(order_reviews["review_answer_timestamp"]==latest_reviews["latest_timestamp"]), "inner")\
    .drop(latest_reviews.order_id, "latest_timestamp")
distinct = order_reviews.select(count_distinct('order_id'), count_distinct('review_id')).show()

# COMMAND ----------

keep = order_reviews.groupBy("review_id").agg(count("order_id").alias("order_id_count"))
keep = keep.where("order_id_count <=1")
order_reviews = order_reviews.join(keep, order_reviews["review_id"]==keep["review_id"],"inner")
order_reviews = order_reviews.drop(keep.review_id, "order_id_count")

distinct = order_reviews.select(count_distinct('order_id'), count_distinct('review_id')).show()

# COMMAND ----------

order_reviews.write.format("csv").option("header","true").mode("overwrite").save("dbfs:/tmp/order_reviews.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating Variables

# COMMAND ----------

# DBTITLE 1,Creating Variables
# 1. Time level variables
##########################

#1.1 Order Processing Time (minutes)
orders = orders.withColumn("order_approved_at_ts", to_timestamp("order_approved_at"))\
    .withColumn("order_purchase_ts", to_timestamp("order_purchase_timestamp"))

orders = orders.withColumn("order_processing_time",(col("order_approved_at_ts").cast("long") - col("order_purchase_ts").cast("long")) / 60)

orders.select("order_processing_time").show(5, False)

# COMMAND ----------

#1.2 Delivery Speed (days)
orders = orders.withColumn("order_delivered_ts", to_timestamp("order_delivered_customer_date"))\
.withColumn("delivery_speed_days",(col("order_delivered_ts").cast("long") - col("order_purchase_ts").cast("long")) / (3600 * 24))

orders.select("delivery_speed_days").show(5, False)

# COMMAND ----------

#1.3 Estimated Delivery Accuracy

orders = orders.withColumn("order_estimated_delivery_ts", to_timestamp("order_estimated_delivery_date"))\
    .withColumn("delivery_accuracy_days", datediff(col("order_delivered_ts"), col("order_estimated_delivery_ts")))

orders.select("delivery_accuracy_days").show(5, False)

# COMMAND ----------

#1.4 Order Month
orders = orders.withColumn("order_month", month(to_timestamp("order_purchase_timestamp")))

# COMMAND ----------

#1.5 Order Quarter
orders = orders.withColumn("order_quarter", quarter(to_timestamp("order_purchase_timestamp")))

# COMMAND ----------

orders = orders.drop('order_approved_at_ts', 'order_purchase_ts', 'order_delivered_ts')
orders.columns

# COMMAND ----------

#2. Customer behavior variables
###############################


#2.1 avg_cust_value: Average payment per order per customer
#2.2 Num_orders: Count of the number of orders made by a customer

#Join orders with order_payments
df = orders.join(order_payments,orders['order_id']==order_payments['order_id'],"leftouter")
df=df.drop(order_payments.order_id)

# COMMAND ----------

cust_data=df.groupBy("customer_id").agg(mean("payment_value").alias("avg_cust_value"),
                                         count("order_id").alias("Num_orders"))
cust_data.show(3)

# COMMAND ----------

#3. Order level variables
##########################

#3.1 Total Item Price

df1=order_items.groupBy("order_id").agg(sum("price").alias("total_item_price"))
df1 = df1.withColumnRenamed("order_id", "df1_id")
orders = orders.join(df1, orders["order_id"] == df1["df1_id"], "left")

# COMMAND ----------

#3.2 Order Item Count

order_items = order_items.withColumnRenamed("order_id", "oi_id")
df = orders.join(order_items, orders["order_id"] == order_items["oi_id"], "left")

df2 = df.groupBy("order_id").agg(max("order_item_id").alias("order_item_count"))
df2 = df2.withColumnRenamed("order_id", "df2_id")
orders = orders.join(df2, orders["order_id"] == df2["df2_id"], "left")

# COMMAND ----------

#3.3 Order Diversity

df = orders.join(order_items, orders["order_id"] == order_items["oi_id"], "left")
df3 = df.groupBy("order_id").agg(count_distinct("product_id").alias("order_diversity"))
df3 = df3.withColumnRenamed("order_id", "df3_id")
orders = orders.join(df3, orders["order_id"] == df3["df3_id"], "left")

# COMMAND ----------

orders = orders.drop('oi_id','df1_id','df2_id','df3_id')

# COMMAND ----------

#3.4 Shipping Cost Percentage
df = orders.join(order_items, orders["order_id"] == order_items["oi_id"], "left")

df4 = df.groupBy("order_id").agg(sum("shipping_cost").alias("total_shipping_cost"))
df4 = df4.withColumnRenamed("order_id", "df4_id")
orders = orders.join(df4, orders["order_id"] == df4["df4_id"], "left")
orders = orders.withColumn('Shipping_Cost_Percentage', col("total_shipping_cost")/(col("total_item_price")+col("total_shipping_cost")))

# COMMAND ----------

orders = orders.drop('total_shipping_cost')
orders.columns

# COMMAND ----------

#3.5 Order status

orders = Pipeline(stages=[
    StringIndexer(inputCol="order_status", outputCol="orderstatusInd", stringOrderType="alphabetAsc"),
    OneHotEncoder(inputCol="orderstatusInd", outputCol="orderstatusInd2")
]).fit(orders).transform(orders).drop("order_status","orderstatusInd")

# COMMAND ----------

#4. Payment level variables
############################

#4.1 Total payment value
#4.2 number of installments
#4.3 number of payments used

df = orders.join(order_payments, orders["order_id"] == order_payments["order_id"], "left")
df = df.drop(order_payments.order_id)
payment_data=df.groupBy('order_id').agg(sum('payment_value').alias('order_payment_value'),
                               max('payment_installments').alias('nbr_installments'),
                               count_distinct('payment_type').alias('nbr_payments_used'))

# COMMAND ----------

#4.4 payment type
payment_type=order_payments.groupBy('order_id').agg(collect_set('payment_type').alias('payment_types_used'))

from pyspark.ml.feature import HashingTF
hashingTF = HashingTF(inputCol="payment_types_used", outputCol="payment_types_dummy")
payment_type=hashingTF.transform(payment_type)

payment_type = payment_type.withColumnRenamed("order_id", "payment_order_id")
payment_data = payment_data.join(payment_type, payment_data["order_id"] == payment_type["payment_order_id"], "left")

payment_data=payment_data.drop("payment_order_id","payment_types_used")
payment_data.show(5)

# COMMAND ----------

missing_value_rows = payment_data.filter(
    " OR ".join([f"{c} IS NULL" for c in payment_data.columns]))

missing_value_rows.show()

# COMMAND ----------

#5. Product level variable
###########################

#5.1 Sum Product Price per order
prod_data=order_items.groupBy("oi_id").agg(sum("price").alias("order_price"))

# COMMAND ----------

#5.2 Total Product category per order 
df = products.join(order_items,products['product_id']==order_items['product_id'],"leftouter")
prod_data1=df.groupBy('oi_id').agg(count_distinct('product_category_name').alias('total_product_category'))
prod_data1.where(col('total_product_category')>1).show(5)

# COMMAND ----------

#5.3 Sum Order Volume: product_length_cm * product_height_cm * product_width_cm
#5.4 Sum Order weight
#5.5 Average product_name_length
#5.6 Average product_description_length

df = products.join(order_items,products['product_id']==order_items['product_id'],"leftouter")
df=df.drop(order_items.product_id)
df = df.withColumn("Volume_cm3",col("product_length_cm")*col("product_height_cm")*col("product_width_cm"))
df.show(3)

# COMMAND ----------

order_data=df.groupBy("oi_id").agg(sum("Volume_cm3").alias("order_volume_cm3"),
                                    (sum("product_weight_g")/1000).alias("order_weight_kg"),
                                    mean("product_name_lenght").alias("avg_prod_name_lenght"),
                                    mean("product_description_lenght").alias("avg_prod_description_lenght"))

# COMMAND ----------

#Create independent variables table
start_date = "2020-09-01"
end_date = "2022-06-30"

indep = orders.where((col("order_purchase_timestamp") >= start_date) & (col("order_purchase_timestamp") <= end_date))
indep = indep.drop('order_purchase_timestamp',
 'order_approved_at',
 'order_delivered_carrier_date',
 'order_delivered_customer_date',
 'order_estimated_delivery_date', 'order_estimated_delivery_ts',
 'order_item_id',
 'product_id',
 'price',
 'shipping_cost',
 'df4_id')

# COMMAND ----------

print(indep.count())
print(orders.count())

# COMMAND ----------

indep=indep.join(cust_data, indep['customer_id']==cust_data['customer_id'],'leftouter')
indep=indep.join(prod_data, indep['order_id']==prod_data['oi_id'],'leftouter') 
indep = indep.drop(prod_data.oi_id)

indep=indep.join(prod_data1, indep['order_id']==prod_data1['oi_id'],'leftouter') 
indep = indep.drop(prod_data1.oi_id)

indep=indep.join(order_data, indep['order_id']==order_data['oi_id'],'leftouter') 
indep=indep.join(payment_data, indep['order_id']==payment_data['order_id'],'leftouter') 

indep = indep.drop(order_data.oi_id, payment_data.order_id, 'customer_id')

# COMMAND ----------

indep = indep.withColumn('check', col('total_item_price')==col('order_price'))
indep.select(col('check')).distinct().show()

# COMMAND ----------

indep = indep.drop('order_price','check')

# COMMAND ----------

#Create basetable
basetable = indep.join(order_reviews, indep['order_id']==order_reviews['order_id'],'inner')
basetable = basetable.drop(order_reviews.order_id, 'review_creation_date', 'review_answer_timestamp')
basetable = basetable.withColumn("review_score", col("review_score").cast("double"))

from pyspark.ml.feature import Binarizer
binarizer = Binarizer(inputCol="review_score", outputCol="label", threshold=3)
basetable = binarizer.transform(basetable).drop('review_score', 'review_id')

# COMMAND ----------

distribution = basetable.groupBy('label').count()
distribution.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Correction and Value Transformation

# COMMAND ----------

# DBTITLE 1,Data Correction and Value Transformation
basetable.select([count(when(col(c).isNull(), c)).alias(c) for c in basetable.columns]).show()

# COMMAND ----------

#drop rows with missing values in payment info
basetable = basetable.dropna(subset=["payment_types_dummy"])
basetable.select([count(when(col(c).isNull(), c)).alias(c) for c in basetable.columns]).show()

# COMMAND ----------

df_train, df_temp = basetable.randomSplit([0.7, 0.3],seed=121)
df_test, df_val= df_temp.randomSplit([0.5, 0.5],seed=121)
print(basetable.count(),df_train.count(),df_test.count(),df_val.count())

# COMMAND ----------

## Missing values
## Replace misisng values of some columns with averages
columns_fillna = ["order_processing_time", "delivery_speed_days", "delivery_accuracy_days","total_item_price","order_item_count","Shipping_Cost_Percentage","total_product_category","order_volume_cm3","order_weight_kg","avg_prod_name_lenght","avg_prod_description_lenght","order_payment_value","nbr_installments", "avg_cust_value"]

# Compute and store the mean for each column
means = {
    col_name: df_train.select(mean(col_name)).collect()[0][0]
    for col_name in columns_fillna}

# Replace null values in each specified column using the computed means
for col_name, mean_value in means.items():
    df_train = df_train.na.fill({col_name: mean_value})

df_train.select([count(when(col(c).isNull(), c)).alias(c) for c in df_train.columns]).show()

# COMMAND ----------

for col_name, mean_value in means.items():
    df_test = df_test.na.fill({col_name: mean_value})
df_test.select([count(when(col(c).isNull(), c)).alias(c) for c in df_test.columns]).show()

# COMMAND ----------

for col_name, mean_value in means.items():
    df_val = df_val.na.fill({col_name: mean_value})
df_val.select([count(when(col(c).isNull(), c)).alias(c) for c in df_val.columns]).show()

# COMMAND ----------

## Outliers
# Get list of numerical columns
numeric_columns = [col for col, dtype in df_train.dtypes if dtype in ('int', 'bigint', 'float', 'double')]
numeric_columns

# COMMAND ----------

numeric_columns = ['order_processing_time','delivery_speed_days','delivery_accuracy_days','total_item_price',
                   'order_item_count','order_diversity','Shipping_Cost_Percentage','avg_cust_value','Num_orders',
                   'total_product_category','order_volume_cm3','order_weight_kg','avg_prod_name_lenght',
                   'avg_prod_description_lenght','order_payment_value','nbr_installments','nbr_payments_used']

# Define the lower and upper percentiles for Winsorization
lower_percentile = 0.01
upper_percentile = 0.99

# Loop through each numerical column and apply Winsorization
for column in numeric_columns:
    # Compute the percentiles for the column
    lower_bound, upper_bound = df_train.approxQuantile(column, [lower_percentile, upper_percentile], 0.01)
    
    # Winsorize the column by clipping values outside the bounds
    df_train = df_train.withColumn(
        column,
        when(col(column) < lower_bound, lower_bound).when(col(column) > upper_bound, upper_bound).otherwise(col(column)))
    
    df_test = df_test.withColumn(
        column,
        when(col(column) < lower_bound, lower_bound).when(col(column) > upper_bound, upper_bound).otherwise(col(column)))
    
    df_val = df_val.withColumn(
        column,
        when(col(column) < lower_bound, lower_bound).when(col(column) > upper_bound, upper_bound).otherwise(col(column)))

# Show the winsorized DataFrame
df_train.show(5)
df_test.show(5)
df_val.show(5)


# COMMAND ----------

df_train.write.mode("overwrite").parquet("dbfs:/tmp/df_train.parquet")
df_test.write.mode("overwrite").parquet("dbfs:/tmp/df_test.parquet")
df_val.write.mode("overwrite").parquet("dbfs:/tmp/df_val.parquet")

# COMMAND ----------

df_train=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_train.parquet")

df_test=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_test.parquet")

df_val=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_val.parquet")

# COMMAND ----------

df_train = df_train.drop("order_id")
df_test = df_test.drop("order_id")
df_val = df_val.drop("order_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Building

# COMMAND ----------

# DBTITLE 1,Model Building
df_train = Pipeline(stages=[
  RFormula(formula="label ~ .")
]).fit(df_train).transform(df_train).select("features","label")
df_train.show(5, truncate=False)

# COMMAND ----------

df_test = Pipeline(stages=[
  RFormula(formula="label ~ .")
]).fit(df_test).transform(df_test).select("features","label")

# COMMAND ----------

df_val=Pipeline(stages=[
  RFormula(formula="label ~ .")
]).fit(df_val).transform(df_val).select("features","label")

# COMMAND ----------

# DBTITLE 1,Choosing the best Parameters
# Random Forest
hyperparams = [
    {"numTrees": 150, "maxDepth": 10},
    {"numTrees": 150, "maxDepth": 15},
    {"numTrees": 200, "maxDepth": 20}
]

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Step 7: Train & Evaluate Models Manually
best_RF = None
best_AUC = 0
best_params = {}

for params in hyperparams:
    # Train model with current hyperparameters
    rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                numTrees=params["numTrees"], maxDepth=params["maxDepth"])
    model = rf.fit(df_train)
    
    # Evaluate on Validation Set
    val_predictions = model.transform(df_val)
    val_AUC = evaluator.evaluate(val_predictions)
    
    print(f"Params: {params}, Validation AUC: {val_AUC:.4f}")
    
    # Select the Best Model
    if val_AUC > best_AUC:
        best_AUC = val_AUC
        best_RF = model
        best_params = params

# COMMAND ----------

# Evaluate the Best Model on the Test Set
RF_predictions = best_RF.transform(df_test)
test_AUC = evaluator.evaluate(RF_predictions)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(RF_predictions)

print(f"\nBest Hyperparameters: {best_params}")
print(f"Test AUC of Best Model: {test_AUC:.4f}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

best_RF.save("dbfs:/tmp/Random_Forest_model")

# COMMAND ----------

# Logistic Regression
hyperparams = [
    {"regParam": 0.1, "maxIter": 50},
    {"regParam": 0.01, "maxIter": 100},
    {"regParam": 0.1, "maxIter": 150}
]

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Step 7: Train & Evaluate Models Manually
best_LR = None
best_AUC = 0
best_params = {}

for params in hyperparams:
    # Train model with current hyperparameters
    lr = LogisticRegression(featuresCol="features", labelCol="label",
                                regParam=params["regParam"], maxIter=params["maxIter"])
    model = lr.fit(df_train)
    
    # Evaluate on Validation Set
    val_predictions = model.transform(df_val)
    val_AUC = evaluator.evaluate(val_predictions)
    
    print(f"Params: {params}, Validation AUC: {val_AUC:.4f}")
    
    # Select the Best Model
    if val_AUC > best_AUC:
        best_AUC = val_AUC
        best_LR = model
        best_params = params

# COMMAND ----------

# Evaluate the Best Model on the Test Set
LR_predictions = best_LR.transform(df_test)
test_AUC = evaluator.evaluate(LR_predictions)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(LR_predictions)

print(f"\nBest Hyperparameters: {best_params}")
print(f"Test AUC of Best Model: {test_AUC:.4f}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

best_LR.save("dbfs:/tmp/logistic_reg_model")

# COMMAND ----------

# DBTITLE 1,Running again using the saved model
LR_model_copy = LogisticRegressionModel.load("dbfs:/tmp/logistic_reg_model")

# COMMAND ----------

# Evaluate the Best Model on the Test Set
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

LR_predictions_copy = LR_model_copy.transform(df_test)
test_AUC = evaluator.evaluate(LR_predictions_copy)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(LR_predictions_copy)

for param in LR_model_copy.extractParamMap():
    print(f"{param.name}: {LR_model_copy.extractParamMap()[param]}")
print(f"Test AUC of Best Model: {test_AUC:.4f}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

# Gradient Boosting
hyperparams = [
    {"maxDepth": 3, "maxIter": 10}
]

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Step 7: Train & Evaluate Models Manually
best_gb = None
best_AUC = 0
best_params = {}

for params in hyperparams:
    # Train model with current hyperparameters
    gb = GBTClassifier(featuresCol="features", labelCol="label",
                                maxDepth=params["maxDepth"], maxIter=params["maxIter"])
    model = gb.fit(df_train)
    
    # Evaluate on Validation Set
    val_predictions = model.transform(df_val)
    val_AUC = evaluator.evaluate(val_predictions)
    
    print(f"Params: {params}, Validation AUC: {val_AUC:.4f}")
    
    # Select the Best Model
    if val_AUC > best_AUC:
        best_AUC = val_AUC
        best_gb = model
        best_params = params

# COMMAND ----------

# Evaluate the Best Model on the Test Set
gb_predictions = best_gb.transform(df_test)
test_AUC = evaluator.evaluate(gb_predictions)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(gb_predictions)

print(f"\nBest Hyperparameters: {best_params}")
print(f"Test AUC of Best Model: {test_AUC:.4f}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------


test_AUC = evaluator.evaluate(gb_predictions)
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(gb_predictions)

print(f"Test AUC of Best Model: {test_AUC:.4f}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

best_gb.save("dbfs:/tmp/gb_model")

# COMMAND ----------

GB_model_copy = GBTClassificationModel.load("dbfs:/tmp/gb_model")

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
gb_predictions_copy = GB_model_copy.transform(df_test)

test_AUC = evaluator.evaluate(gb_predictions_copy)
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(gb_predictions_copy)
for param in GB_model_copy.extractParamMap():
    print(f"{param.name}: {GB_model_copy.extractParamMap()[param]}")
print(f"Test AUC of Best Model: {test_AUC:.4f}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

predictions = GB_model_copy.transform(df_test)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
test_AUC = evaluator.evaluate(predictions)

# Extract probability column (as an array) and get probability of class 1
predictions = predictions.withColumn("probability", vector_to_array("probability")[1])

# Compute ROC curve manually using Spark's built-in `roc` function
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create an evaluator and get the ROC curve data
roc_curve = predictions.select("label", "probability").rdd.map(lambda row: (float(row["probability"]), float(row["label"])))

# Sort by probability descending for proper ROC calculation
roc_curve = sorted(roc_curve.collect(), key=lambda x: -x[0])


tpr = []  # True Positive Rate
fpr = []  # False Positive Rate
thresholds = []  # Threshold values

pos_total = sum([x[1] for x in roc_curve])  # Total positives
neg_total = len(roc_curve) - pos_total  # Total negatives
tp = 0  # True positives count
fp = 0  # False positives count

for prob, label in roc_curve:
    if label == 1:
        tp += 1
    else:
        fp += 1
    tpr.append(tp / pos_total)
    fpr.append(fp / neg_total)
    thresholds.append(prob)

roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})

# COMMAND ----------

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(roc_df["FPR"], roc_df["TPR"], label=f"AUC = {test_AUC:.4f}", color='blue')
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal Line for Random Classifier
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for Gradient Boosting Model")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# COMMAND ----------

# Get labels & predictions
confusion_matrix = predictions.groupBy("label", "prediction").count()

# Show matrix
confusion_matrix.show()

# Convert Spark DataFrame to Pandas
conf_matrix_pd = confusion_matrix.toPandas()

# Create a pivot table for plotting
conf_matrix_pivot = conf_matrix_pd.pivot(index="label", columns="prediction", values="count").fillna(0)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_pivot, annot=True, fmt="g", cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix for GBT Model")
plt.show()

# COMMAND ----------

#Get feature importances
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
ExtractFeatureImp(GB_model_copy.featureImportances, df_test, "features").head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final Prediction

# COMMAND ----------

# DBTITLE 1,Final Prediction
#1.1 Order Processing Time (minutes)
test_orders = test_orders.withColumn("order_approved_at_ts", to_timestamp("order_approved_at"))\
    .withColumn("order_purchase_ts", to_timestamp("order_purchase_timestamp"))
test_orders = test_orders.withColumn("order_processing_time",(col("order_approved_at_ts").cast("long") - col("order_purchase_ts").cast("long")) / 60)

#1.2 Delivery Speed (days)
test_orders = test_orders.withColumn("order_delivered_ts", to_timestamp("order_delivered_customer_date"))\
.withColumn("delivery_speed_days",(col("order_delivered_ts").cast("long") - col("order_purchase_ts").cast("long")) / (3600 * 24))

#1.3 Estimated Delivery Accuracy
test_orders = test_orders.withColumn("order_estimated_delivery_ts", to_timestamp("order_estimated_delivery_date"))\
    .withColumn("delivery_accuracy_days", datediff(col("order_delivered_ts"), col("order_estimated_delivery_ts")))

#1.4 Order Month
test_orders = test_orders.withColumn("order_month", month(to_timestamp("order_purchase_timestamp")))

#1.5 Order Quarter
test_orders = test_orders.withColumn("order_quarter", quarter(to_timestamp("order_purchase_timestamp")))
test_orders = test_orders.drop('order_approved_at_ts', 'order_purchase_ts', 'order_delivered_ts')

#2.1 avg_cust_value: Average payment per order per customer
#2.2 Num_test_orders: Count of the number of test_orders made by a customer
df = test_orders.join(test_order_payments,test_orders['order_id']==test_order_payments['order_id'],"leftouter")
df=df.drop(test_order_payments.order_id)

cust_data=df.groupBy("customer_id").agg(mean("payment_value").alias("avg_cust_value"),
                                         count("order_id").alias("Num_test_orders"))

#3.1 Total Item Price
df1=test_order_items.groupBy("order_id").agg(sum("price").alias("total_item_price"))
df1 = df1.withColumnRenamed("order_id", "df1_id")
test_orders = test_orders.join(df1, test_orders["order_id"] == df1["df1_id"], "left")

#3.2 Order Item Count
test_order_items = test_order_items.withColumnRenamed("order_id", "oi_id")
df = test_orders.join(test_order_items, test_orders["order_id"] == test_order_items["oi_id"], "left")

df2 = df.groupBy("order_id").agg(max("order_item_id").alias("order_item_count"))
df2 = df2.withColumnRenamed("order_id", "df2_id")
test_orders = test_orders.join(df2, test_orders["order_id"] == df2["df2_id"], "left")

#3.3 Order Diversity
df = test_orders.join(test_order_items, test_orders["order_id"] == test_order_items["oi_id"], "left")
df3 = df.groupBy("order_id").agg(count_distinct("product_id").alias("order_diversity"))
df3 = df3.withColumnRenamed("order_id", "df3_id")
test_orders = test_orders.join(df3, test_orders["order_id"] == df3["df3_id"], "left")
test_orders = test_orders.drop('oi_id','df1_id','df2_id','df3_id')

#3.4 Shipping Cost Percentage
df = test_orders.join(test_order_items, test_orders["order_id"] == test_order_items["oi_id"], "left")
df4 = df.groupBy("order_id").agg(sum("shipping_cost").alias("total_shipping_cost"))
df4 = df4.withColumnRenamed("order_id", "df4_id")
test_orders = test_orders.join(df4, test_orders["order_id"] == df4["df4_id"], "left")
test_orders = test_orders.withColumn('Shipping_Cost_Percentage', col("total_shipping_cost")/(col("total_item_price")+col("total_shipping_cost")))
test_orders = test_orders.drop('total_shipping_cost')

#3.5 Order status
test_orders = Pipeline(stages=[
    StringIndexer(inputCol="order_status", outputCol="orderstatusInd", stringOrderType="alphabetAsc"),
    OneHotEncoder(inputCol="orderstatusInd", outputCol="orderstatusInd2")
]).fit(orders).transform(test_orders).drop("order_status","orderstatusInd")

#4.1 Total payment value
#4.2 number of installments
#4.3 number of payments used
df = test_orders.join(test_order_payments, test_orders["order_id"] == test_order_payments["order_id"], "left")
df = df.drop(test_order_payments.order_id)
payment_data=df.groupBy('order_id').agg(sum('payment_value').alias('order_payment_value'),
                               max('payment_installments').alias('nbr_installments'),
                               count_distinct('payment_type').alias('nbr_payments_used'))

#4.4 payment type
payment_type=test_order_payments.groupBy('order_id').agg(collect_set('payment_type').alias('payment_types_used'))
hashingTF = HashingTF(inputCol="payment_types_used", outputCol="payment_types_dummy")
payment_type=hashingTF.transform(payment_type)

payment_type = payment_type.withColumnRenamed("order_id", "payment_order_id")
payment_data = payment_data.join(payment_type, payment_data["order_id"] == payment_type["payment_order_id"], "left")

payment_data=payment_data.drop("payment_order_id","payment_types_used")

#5.1 Sum Product Price per order
prod_data=test_order_items.groupBy("oi_id").agg(sum("price").alias("order_price"))

#5.2 Total Product category per order 
df = test_products.join(test_order_items,test_products['product_id']==test_order_items['product_id'],"leftouter")
prod_data1=df.groupBy('oi_id').agg(count_distinct('product_category_name').alias('total_product_category'))
prod_data1.where(col('total_product_category')>1)

#5.3 Sum Order Volume: product_length_cm * product_height_cm * product_width_cm
#5.4 Sum Order weight
#5.5 Average product_name_length
#5.6 Average product_description_length
df = test_products.join(test_order_items,test_products['product_id']==test_order_items['product_id'],"leftouter")
df=df.drop(test_order_items.product_id)
df = df.withColumn("Volume_cm3",col("product_length_cm")*col("product_height_cm")*col("product_width_cm"))

order_data=df.groupBy("oi_id").agg(sum("Volume_cm3").alias("order_volume_cm3"),
                                    (sum("product_weight_g")/1000).alias("order_weight_kg"),
                                    mean("product_name_lenght").alias("avg_prod_name_lenght"),
                                    mean("product_description_lenght").alias("avg_prod_description_lenght"))

# COMMAND ----------

#Create independent_test variables table
start_date = "2022-07-01"
end_date = "2022-09-30"

indep_test = test_orders.where((col("order_purchase_timestamp") >= start_date) & (col("order_purchase_timestamp") <= end_date))
indep_test = indep_test.drop('order_purchase_timestamp',
 'order_approved_at',
 'order_delivered_carrier_date',
 'order_delivered_customer_date',
 'order_estimated_delivery_date', 'order_estimated_delivery_ts',
 'order_item_id',
 'product_id',
 'price',
 'shipping_cost',
 'df4_id')

indep_test=indep_test.join(cust_data, indep_test['customer_id']==cust_data['customer_id'],'leftouter')
indep_test=indep_test.join(prod_data, indep_test['order_id']==prod_data['oi_id'],'leftouter') 
indep_test = indep_test.drop(prod_data.oi_id)

indep_test=indep_test.join(prod_data1, indep_test['order_id']==prod_data1['oi_id'],'leftouter') 
indep_test = indep_test.drop(prod_data1.oi_id)

indep_test=indep_test.join(order_data, indep_test['order_id']==order_data['oi_id'],'leftouter') 
indep_test=indep_test.join(payment_data, indep_test['order_id']==payment_data['order_id'],'leftouter') 

indep_test = indep_test.drop(order_data.oi_id, payment_data.order_id, 'customer_id')

# COMMAND ----------

indep_test = indep_test.drop('order_price')

# COMMAND ----------

indep_test.select([count(when(col(c).isNull(), c)).alias(c) for c in indep_test.columns]).show()

# COMMAND ----------

#Replacing missing values
columns_fillna = ["order_processing_time", "delivery_speed_days", "delivery_accuracy_days","total_item_price","order_item_count","Shipping_Cost_Percentage","total_product_category","order_volume_cm3","order_weight_kg","avg_prod_name_lenght","avg_prod_description_lenght","order_payment_value","nbr_installments", "avg_cust_value"]

# Compute and store the mean for each column
means = {
    col_name: indep_test.select(mean(col_name)).collect()[0][0]
    for col_name in columns_fillna}

# Replace null values in each specified column using the computed means
for col_name, mean_value in means.items():
    indep_test = indep_test.na.fill({col_name: mean_value})

indep_test.select([count(when(col(c).isNull(), c)).alias(c) for c in indep_test.columns]).show()

# COMMAND ----------

## Outliers
# Get list of numerical columns
numeric_columns = [col for col, dtype in indep_test.dtypes if dtype in ('int', 'bigint', 'float', 'double')]
numeric_columns

# COMMAND ----------

numeric_columns = ['order_processing_time','delivery_speed_days','delivery_accuracy_days','total_item_price',
                   'order_item_count','order_diversity','Shipping_Cost_Percentage','avg_cust_value','Num_test_orders',
                   'total_product_category','order_volume_cm3','order_weight_kg','avg_prod_name_lenght',
                   'avg_prod_description_lenght','order_payment_value','nbr_installments','nbr_payments_used']

# Define the lower and upper percentiles for Winsorization
lower_percentile = 0.01
upper_percentile = 0.99

# Loop through each numerical column and apply Winsorization
for column in numeric_columns:
    # Compute the percentiles for the column
    lower_bound, upper_bound = indep_test.approxQuantile(column, [lower_percentile, upper_percentile], 0.01)
    
    # Winsorize the column by clipping values outside the bounds
    indep_test = indep_test.withColumn(
        column,
        when(col(column) < lower_bound, lower_bound).when(col(column) > upper_bound, upper_bound).otherwise(col(column)))

# Show the winsorized DataFrame
indep_test.show()

# COMMAND ----------

indep_test.write.mode("overwrite").parquet("dbfs:/tmp/indep_test.parquet")

# COMMAND ----------

indep_test=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/indep_test.parquet")

# COMMAND ----------

indep_test.columns

# COMMAND ----------

#VectorAssembler: combine multiple numeric features in one big vector.
va = VectorAssembler(inputCols=['order_processing_time','delivery_speed_days',
 'delivery_accuracy_days',
 'order_month',
 'order_quarter',
 'total_item_price',
 'order_item_count',
 'order_diversity',
 'Shipping_Cost_Percentage',
 'orderstatusInd2',
 'avg_cust_value',
 'Num_test_orders',
 'total_product_category',
 'order_volume_cm3',
 'order_weight_kg',
 'avg_prod_name_lenght',
 'avg_prod_description_lenght',
 'order_payment_value',
 'nbr_installments',
 'nbr_payments_used',
 'payment_types_dummy'],outputCol="features")
vaDF = va.transform(indep_test)

indep_test_model= vaDF.select('order_id','features')
indep_test_model.show(5,False) 

# COMMAND ----------

indep_test_model.columns

# COMMAND ----------

# DBTITLE 1,Loading Gradient Boosting model
# Load the trained logistic regression model
GBT_model_copy = GBTClassificationModel.load("dbfs:/tmp/gb_model")

# COMMAND ----------

final_prediction=GBT_model_copy.transform(indep_test_model)
final_prediction.show(5)

# COMMAND ----------

# DBTITLE 1,Saving the result
result=final_prediction.select('order_id','prediction')
result.write.format("csv").option("header","true").mode("overwrite").save("dbfs:/tmp/result.csv")

# COMMAND ----------

file_to_dwnld=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load('dbfs:/tmp/result.csv')

file_to_dwnld.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### MultiClass Classification

# COMMAND ----------

# DBTITLE 1,Loading Data
df_train=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_train.parquet")

df_val=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_val.parquet")

df_test=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_test.parquet")


order_reviews=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/order_reviews.csv")

# COMMAND ----------

df_train = df_train.join(order_reviews, df_train['order_id']==order_reviews['order_id'],'inner')
df_train = df_train.drop(order_reviews.order_id, "label", "review_creation_date", "review_answer_timestamp", "review_id")
df_train = df_train.withColumnRenamed("review_score", "label")

df_val = df_val.join(order_reviews, df_val['order_id']==order_reviews['order_id'],'inner')
df_val = df_val.drop(order_reviews.order_id, "label", "review_creation_date", "review_answer_timestamp", "review_id")
df_val = df_val.withColumnRenamed("review_score", "label")

df_test = df_test.join(order_reviews, df_test['order_id']==order_reviews['order_id'],'inner')
df_test = df_test.drop(order_reviews.order_id, "label", "review_creation_date", "review_answer_timestamp", "review_id")
df_test = df_test.withColumnRenamed("review_score", "label")

# COMMAND ----------

indep_test=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/indep_test.parquet")

# COMMAND ----------

df_train = Pipeline(stages=[
  RFormula(formula="label ~ . -order_id")
]).fit(df_train).transform(df_train).select("features","label")

df_val = Pipeline(stages=[
  RFormula(formula="label ~ . -order_id")
]).fit(df_val).transform(df_val).select("features","label")

df_test = Pipeline(stages=[
  RFormula(formula="label ~ . -order_id")
]).fit(df_test).transform(df_test).select("features","label")

# COMMAND ----------

# Random Forest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

hyperparams = [
    {"numTrees": 150, "maxDepth": 10},
    {"numTrees": 150, "maxDepth": 15},
    {"numTrees": 200, "maxDepth": 20}
]

evaluator = MulticlassClassificationEvaluator(labelCol="label")

# Step 7: Train & Evaluate Models Manually
best_RF = None
best_accuracy = 0
best_params = {}

for params in hyperparams:
    # Train model with current hyperparameters
    rf = RandomForestClassifier(featuresCol="features", labelCol="label",weightCol="weights",
                                numTrees=params["numTrees"], maxDepth=params["maxDepth"])
    model = rf.fit(df_train)
    
    # Evaluate on Validation Set
    val_predictions = model.transform(df_val)
    val_accuracy = evaluator.evaluate(val_predictions)
    
    print(f"Params: {params}, Validation accuracy: {val_accuracy:.4f}")
    
    # Select the Best Model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_RF = model
        best_params = params


# COMMAND ----------

RF_predictions = best_RF.transform(df_test)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(RF_predictions)

print(f"\nBest Hyperparameters: {best_params}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

best_RF.write().overwrite().save("dbfs:/tmp/random_forest_model_multi")

# COMMAND ----------

# DBTITLE 1,Model Building
# Logistic Regression
hyperparams = [
    {"regParam": 0.01, "maxIter": 50},
    {"regParam": 0.01, "maxIter": 100},
    {"regParam": 0.1, "maxIter": 100}
]

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

best_LR = None
best_accuracy = 0
best_params = {}

for params in hyperparams:
    lr = LogisticRegression(featuresCol="features", labelCol="label",
                                regParam=params["regParam"], maxIter=params["maxIter"])
    model = lr.fit(df_train)
    
    # Evaluate on Validation Set
    val_predictions = model.transform(df_val)
    val_accuracy = evaluator.evaluate(val_predictions)
    
    print(f"Params: {params}, Validation accuracy: {val_accuracy:.4f}")
    
    # Select the Best Model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_LR = model
        best_params = params

# COMMAND ----------

# Evaluate the Best Model on the Test Set
LR_predictions = best_LR.transform(df_test)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(LR_predictions)

print(f"\nBest Hyperparameters: {best_params}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

best_LR.write().overwrite().save("dbfs:/tmp/logistic_reg_model_multi")

# COMMAND ----------

# Load trained multiclass Logistic Regression model
LR_multi_model_copy = LogisticRegressionModel.load("dbfs:/tmp/logistic_reg_model_multi")

def ExtractFeatureImp_LR_Multiclass(coefficients_matrix, dataset, featuresCol):
    feature_metadata = dataset.schema[featuresCol].metadata.get("ml_attr", {}).get("attrs", {})

    if not feature_metadata:
        raise ValueError(f"Feature metadata not found in column '{featuresCol}'. Ensure it was created using VectorAssembler.")

    feature_list = []
    for feature_type in feature_metadata:
        feature_list.extend(feature_metadata[feature_type])
    feature_df = pd.DataFrame(feature_list)

    if "idx" not in feature_df.columns:
        raise ValueError("Feature metadata does not contain 'idx'. Check the feature column format.")

    feature_df["idx"] = feature_df["idx"].astype(int)

    coef_matrix = coefficients_matrix.toArray()
    importance_values = np.abs(coef_matrix[:, feature_df["idx"].to_numpy()].mean(axis=0))
    feature_df["importance"] = importance_values.tolist()

    return feature_df.sort_values("importance", ascending=False)

ExtractFeatureImp_LR_Multiclass(LR_multi_model_copy.coefficientMatrix, df_test, "features").head(10)

# COMMAND ----------

scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df_train) 

df_train_scaled = scaler_model.transform(df_train) 
df_val_scaled = scaler_model.transform(df_val)     
df_test_scaled = scaler_model.transform(df_test)

# COMMAND ----------

# Nive Bayes
smoothing_values = [0.5, 1.0, 1.5]
model_types = ["multinomial"]
hyperparams = [{"smoothing": s, "modelType": m} for s in smoothing_values for m in model_types]

evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

best_NB = None
best_accuracy = 0
best_params = {}


for params in hyperparams:
    nb = NaiveBayes(featuresCol="scaled_features", labelCol="label",
                    smoothing=params["smoothing"], modelType=params["modelType"])
    model = nb.fit(df_train_scaled)

    val_predictions = model.transform(df_val_scaled)
    val_accuracy = evaluator.evaluate(val_predictions)

    print(f"Params: {params}, Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_NB = model
        best_params = params

# COMMAND ----------

# Evaluate the Best Model on the Test Set
NB_predictions = best_NB.transform(df_test_scaled)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(NB_predictions)

print(f"\nBest Hyperparameters: {best_params}")
print(f"Test accuracy of Best Model: {accuracy:.4f}")

# COMMAND ----------

# DBTITLE 1,Final Prediction
LR_multi_model_copy = LogisticRegressionModel.load("dbfs:/tmp/logistic_reg_model_multi")
multi_prediction=LR_multi_model_copy.transform(indep_test_model)

multi_prediction.show(5)

# COMMAND ----------

multi_result=multi_prediction.select('order_id','prediction')

# COMMAND ----------

# DBTITLE 1,Merging the binary results
binary_result=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/result.csv")

# COMMAND ----------

binary_result=binary_result.withColumnRenamed('prediction','binary_prediction')
multi_result=multi_result.withColumnRenamed('order_id','order_id1')
multi_result=multi_result.withColumnRenamed('prediction','multiclass_prediction')
final_result=binary_result.join(multi_result, multi_result['order_id1']==binary_result['order_id'], 'leftouter')
final_result=final_result.drop('order_id1')
final_result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Insights

# COMMAND ----------

df_train=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_train.parquet")

df_val=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_val.parquet")

df_test=spark\
.read\
.format("parquet")\
.option("header","true")\
.option("inferSchema","true")\
.load("dbfs:/tmp/df_test.parquet")

orders=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load(filePath1)

# COMMAND ----------

print(df_train.count())
print(df_val.count())
print(df_test.count())

# COMMAND ----------

basetable = df_train.union(df_val).union(df_test)
basetable.count()

# COMMAND ----------

orders1 = orders.select("order_id","order_status")

basetable = basetable.join(orders1, basetable["order_id"] == orders1["order_id"], "left")
basetable = basetable.drop(orders1.order_id)

basetable

# COMMAND ----------

# Order Processing Time
order_processing = basetable.groupBy("label").avg("order_processing_time")
order_processing = order_processing.toPandas()

plt.figure(figsize=(8, 8))
ax = sns.barplot(x='label', y='avg(order_processing_time)', data=order_processing, palette="BuPu", alpha = 0.6)

plt.title("Average Order Processing Time by Label (0 and 1)")
plt.ylabel("Average Order Processing Time")
plt.xlabel("")
plt.xticks(ticks=[0, 1], labels=['Label 0: 1-3(review score)', 'Label 1: 4-5(review score)'])

for container in ax.containers:
    ax.bar_label(container, labels=[f'{height.get_height():.2f}' for height in container], padding=5)
# Reference: https://www.statology.org/pandas-annotate-bar-plot/
plt.show()

# COMMAND ----------

label0 = basetable.where("label == 0")
label1 = basetable.where("label == 1")

avg_0 = label0.agg(avg("order_processing_time")).collect()[0][0]
avg_1 = label1.agg(avg("order_processing_time")).collect()[0][0]
difference = (avg_0 - avg_1)/60
difference

# COMMAND ----------

# delivery_speed_days
delivery_speed = basetable.groupBy("label").avg("delivery_speed_days")
delivery_speed = delivery_speed.toPandas()

plt.figure(figsize=(8, 8))
ax = sns.barplot(x='label', y='avg(delivery_speed_days)', data=delivery_speed, palette="BuPu", alpha = 0.6)

plt.title("Average Delivery Speed Days by Label (0 and 1)")
plt.ylabel("Average Delivery Speed Days")
plt.xlabel("")
plt.xticks(ticks=[0, 1], labels=['Label 0: 1-3(review score)', 'Label 1: 4-5(review score)'])

for container in ax.containers:
    ax.bar_label(container, labels=[f'{height.get_height():.2f}' for height in container], padding=5)

plt.show()

# COMMAND ----------

label0 = basetable.where("label == 0")
label1 = basetable.where("label == 1")

avg_0 = label0.agg(avg("delivery_speed_days")).collect()[0][0]
avg_1 = label1.agg(avg("delivery_speed_days")).collect()[0][0]
difference = avg_0 - avg_1
difference

# COMMAND ----------

# delivery_accuracy_days
delivery_accuracy = basetable.groupBy("label").avg("delivery_accuracy_days")
delivery_accuracy = delivery_accuracy.toPandas()

plt.figure(figsize=(8, 8))
ax = sns.barplot(x='label', y='avg(delivery_accuracy_days)', data=delivery_accuracy, palette="BuPu", alpha = 0.6)

plt.title("Average Delivery Accuracy Days by Label (0 and 1)")
plt.ylabel("Average Delivery Accuracy Days")
plt.xlabel("")
plt.xticks(ticks=[0, 1], labels=['Label 0: 1-3(review score)', 'Label 1: 4-5(review score)'])

for container in ax.containers:
    ax.bar_label(container, labels=[f'{height.get_height():.2f}' for height in container], padding=5)

plt.show()

# COMMAND ----------

label0 = basetable.where("label == 0")
label1 = basetable.where("label == 1")

avg_0 = label0.agg(avg("delivery_accuracy_days")).collect()[0][0]
avg_1 = label1.agg(avg("delivery_accuracy_days")).collect()[0][0]
difference = avg_0 - avg_1
difference

# COMMAND ----------

# Order status
distribution_table0 = label0.groupBy("order_status").count().orderBy("count", ascending=False)
distribution_table1 = label1.groupBy("order_status").count().orderBy("count", ascending=False)

print("Distribution Table for Label 0:")
distribution_table0.show()

print("Distribution Table for Label 1:")
distribution_table1.show()

# COMMAND ----------

# Shipping_Cost_Percentage
shipping_cost = basetable.groupBy("label").avg("Shipping_Cost_Percentage")
shipping_cost = shipping_cost.toPandas()

plt.figure(figsize=(8, 8))
ax = sns.barplot(x='label', y='avg(Shipping_Cost_Percentage)', data=shipping_cost, palette="Purples", alpha = 0.8)

plt.title("Average Shipping Cost Percentage by Label (0 and 1)")
plt.ylabel("Average Shipping Cost Percentage")
plt.xlabel("")
plt.xticks(ticks=[0, 1], labels=['Label 0: 1-3(review score)', 'Label 1: 4-5(review score)'])

for container in ax.containers:
    ax.bar_label(container, labels=[f'{height.get_height():.2f}' for height in container], padding=5)
# Reference: https://www.statology.org/pandas-annotate-bar-plot/
plt.show()

# COMMAND ----------

# order_payment_value
order_payment = basetable.groupBy("label").avg("order_payment_value")
order_payment = order_payment.toPandas()

plt.figure(figsize=(8, 8))
ax = sns.barplot(x='label', y='avg(order_payment_value)', data=order_payment, palette="Purples", alpha = 0.8)

plt.title("Average Order Payment Value by Label (0 and 1)")
plt.ylabel("Average Order Payment Value")
plt.xlabel("")
plt.xticks(ticks=[0, 1], labels=['Label 0: 1-3(review score)', 'Label 1: 4-5(review score)'])

for container in ax.containers:
    ax.bar_label(container, labels=[f'{height.get_height():.2f}' for height in container], padding=5)
# Reference: https://www.statology.org/pandas-annotate-bar-plot/
plt.show()

# COMMAND ----------

# nbr_installments
nbr_installments = basetable.groupBy("label").avg("nbr_installments")
nbr_installments = nbr_installments.toPandas()

plt.figure(figsize=(8, 8))
ax = sns.barplot(x='label', y='avg(nbr_installments)', data=nbr_installments, palette="Purples", alpha = 0.8)

plt.title("Average Nbr of Installments by Label (0 and 1)")
plt.ylabel("Average Nbr of Installments")
plt.xlabel("")
plt.xticks(ticks=[0, 1], labels=['Label 0: 1-3(review score)', 'Label 1: 4-5(review score)'])

for container in ax.containers:
    ax.bar_label(container, labels=[f'{height.get_height():.2f}' for height in container], padding=5)
# Reference: https://www.statology.org/pandas-annotate-bar-plot/
plt.show()

# COMMAND ----------

# Payment type used
payment_types = order_payments.select("order_id","payment_type")
payment = basetable.join(payment_types, basetable["order_id"]==payment_types["order_id"], "left")
payment = payment.drop(payment_types.order_id)

# COMMAND ----------

payment_type_counts = (
    payment.groupBy("label", "payment_type")
    .count()
    .orderBy("label", "count", ascending=False))

payment_type_counts1 = payment_type_counts.groupBy("label").agg(sum("count").alias("total_count"))
payment_type_counts = payment_type_counts.join(payment_type_counts1, on="label")

payment_type_counts = payment_type_counts.withColumn(
    "percentage", round((col("count") / col("total_count") * 100),2))

payment_type_counts = payment_type_counts.orderBy("label", "percentage", ascending=False)
payment_type_counts.select("label", "payment_type", "count", "percentage").show()

# COMMAND ----------

df = products.join(order_items,products['product_id']==order_items['product_id'],"leftouter")
df=df.drop(order_items.product_id)
order_data=df.groupBy("oi_id").agg(mean("product_photos_qty").alias("avg_product_photos_qty"))

basetable = basetable.join(order_data, basetable["order_id"]==order_data["oi_id"])
basetable = basetable.drop(order_data.oi_id)

# COMMAND ----------

# Product Info
pro_info_by_label = basetable.groupBy("label").agg(
    round(avg("avg_prod_name_lenght"),2).alias("avg_prod_name_lenght"),
    round(avg("avg_prod_description_lenght"),2).alias("avg_prod_description_lenght"),
    round(avg("avg_product_photos_qty"),2).alias("avg_product_photos_qty"))

pro_info_by_label.show()

# COMMAND ----------

numeric_columns = ['order_processing_time','delivery_speed_days','delivery_accuracy_days','total_item_price',
                   'order_item_count','order_diversity','Shipping_Cost_Percentage','avg_cust_value','Num_orders',
                   'total_product_category','order_volume_cm3','order_weight_kg','avg_prod_name_lenght',
                   'avg_prod_description_lenght','order_payment_value','nbr_installments','nbr_payments_used']
correlations = {}
for feature in numeric_columns:
    correlation = basetable.stat.corr(feature, "label")
    correlations[feature] = correlation

correlation_rows = [Row(feature=key, correlation=value) for key, value in correlations.items()]
correlation_df = basetable.sparkSession.createDataFrame(correlation_rows)

correlation_df = correlation_df.orderBy("correlation", ascending=False)
correlation_df

# COMMAND ----------

correlation_df.show(truncate=False)
