# BigData

Khởi động cum Hadoop bằng 2 câu lệnh
start-yarn.sh
start-dfs.sh
Sau khi khởi động kiểm tra xem đã hoạt động chưa bằng lệnh jps

Cần kích hoạt môi trường trước khi chạy code bằng câu lệnh
source "tên môi trường"/bin/activate

Sau khi đã kích hoạt môi trường thành công, ta bắt đầu chạy lệnh
spark-submit weather_model.py

Sau khi code chạy xong, ta sẽ được 4 file ảnh bao gồm 
- correlation_heatmap.png
- residual_lr.png
- temperature_trend.png
- model_comparison.png

Các file ảnh sẽ được lưu ngay trong thư mục chạy code, và kết quả sẽ được lưu lại trên HDFS
