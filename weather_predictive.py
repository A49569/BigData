from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, avg, stddev, count, year, month, hour, to_timestamp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. TẠO SPARK SESSION
spark = SparkSession.builder \
    .appName("WeatherPredictiveModelAdvanced") \
    .getOrCreate()

# 2. ĐỌC DỮ LIỆU VÀ TIỀN XỬ LÝ
file_path = "hdfs://localhost:9000/weather_data/weatherHistory.csv"
print("Đọc dữ liệu từ:", file_path)
df = spark.read.option("header", "true").csv(file_path, inferSchema=True)

# Đổi tên cột và lọc dữ liệu hợp lệ
df_cleaned = df.select(
    col("Formatted Date").alias("date"),
    col("Temperature (C)").alias("temperature"),
    col("Humidity").alias("humidity"),
    col("Precip Type").alias("precipitation")
).filter(
    (col("temperature").isNotNull()) &
    (col("humidity").isNotNull()) &
    (col("precipitation").isNotNull()) &
    (col("temperature").between(-50, 50)) &
    (col("humidity").between(0, 1))
)

indexer = StringIndexer(inputCol="precipitation", outputCol="precipitation_index")
df_cleaned = indexer.fit(df_cleaned).transform(df_cleaned)

# Thêm thông tin thời gian
df_cleaned = df_cleaned.withColumn("timestamp", to_timestamp("date")) \
    .withColumn("year", year("timestamp")) \
    .withColumn("month", month("timestamp")) \
    .withColumn("hour", hour("timestamp"))

# Lưu vào HDFS
output_path = "hdfs://localhost:9000/weather_data/cleaned_weather_data"
df_cleaned.write.mode("overwrite").parquet(output_path)
df = spark.read.parquet(output_path)
print("✅ Dữ liệu đã lưu và đọc lại thành công từ:", output_path)

# 3. PHÂN TÍCH NÂNG CAO
print("\n📊 Thống kê tổng quát:")
df.groupBy().agg(
    avg("temperature").alias("avg_temp"),
    stddev("temperature").alias("std_temp"),
    avg("humidity").alias("avg_humidity"),
    avg("precipitation_index").alias("avg_precipitation"),
    count("*").alias("record_count")
).show()

print("\n📊 Nhiệt độ trung bình theo tháng:")
df.groupBy("year", "month").agg(avg("temperature").alias("avg_temp")).orderBy("year", "month").show(12)

print("\n📊 So sánh nhiệt độ và độ ẩm theo loại mưa:")
df.groupBy("precipitation").agg(avg("temperature"), avg("humidity")).show()

# 4. HUẤN LUYỆN MÔ HÌNH
assembler = VectorAssembler(inputCols=["humidity", "precipitation_index"], outputCol="features")
df = assembler.transform(df)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

lr = LinearRegression(featuresCol="features", labelCol="temperature")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.evaluate(test_data)
print(f"Linear Regression RMSE: {lr_predictions.rootMeanSquaredError}")
print(f"Linear Regression R2: {lr_predictions.r2}")
rf = RandomForestRegressor(featuresCol="features", labelCol="temperature", numTrees=100)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="temperature", metricName="rmse")
rf_rmse = evaluator.evaluate(rf_predictions)
rf_r2 = evaluator.evaluate(rf_predictions, {evaluator.metricName: "r2"})
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Random Forest R2: {rf_r2}")

# 5. DỰ ĐOÁN & TRỰC QUAN HÓA NÂNG CAO
future_data = pd.DataFrame({
    'humidity': [0.6, 0.65],
    'precipitation_index': [0.0, 1.0]
})
future_df = spark.createDataFrame(future_data)
future_df = assembler.transform(future_df)
predictions = lr_model.transform(future_df)
predictions.select("prediction").show()

# TRỰC QUAN HÓA
lr_result = lr_model.transform(test_data).select("temperature", "prediction")
lr_result_pd = lr_result.withColumn("residual", col("temperature") - col("prediction")).toPandas()
rf_result_pd = rf_predictions.select("temperature", "prediction").toPandas()

# Heatmap tương quan
sample_corr = df.select("temperature", "humidity", "precipitation_index").sample(False, 0.05).toPandas()
plt.figure(figsize=(5, 4))
sns.heatmap(sample_corr.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")


# Phân phối sai số Linear Regression
plt.figure()
sns.histplot(lr_result_pd["residual"], kde=True)
plt.title("Residual Distribution (Linear Regression)")
plt.xlabel("Residual")
plt.savefig("residual_lr.png")
plt.show()

# Biểu đồ nhiệt độ theo thời gian
plot_pd = df.select("timestamp", "temperature").orderBy("timestamp").limit(1000).toPandas()
plt.figure(figsize=(10, 4))
sns.lineplot(x="timestamp", y="temperature", data=plot_pd)
plt.title("Temperature Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("temperature_trend.png")
plt.show()

# So sánh mô hình
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=lr_result_pd, x="temperature", y="prediction")
plt.plot([lr_result_pd["temperature"].min(), lr_result_pd["temperature"].max()],
         [lr_result_pd["temperature"].min(), lr_result_pd["temperature"].max()], 'r--')
plt.title("Linear Regression")

plt.subplot(1, 2, 2)
sns.scatterplot(data=rf_result_pd, x="temperature", y="prediction")
plt.plot([rf_result_pd["temperature"].min(), rf_result_pd["temperature"].max()],
         [rf_result_pd["temperature"].min(), rf_result_pd["temperature"].max()], 'r--')
plt.title("Random Forest")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# KẾT THÚC
spark.stop()