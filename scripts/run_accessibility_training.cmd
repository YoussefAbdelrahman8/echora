@echo off
setlocal
cd /d C:\Users\TAMER\PycharmProjects\echora

"C:\Users\TAMER\PycharmProjects\echora\.venv\Scripts\python.exe" scripts\train_indoor_yolo.py ^
  --dataset datasets\accessibility_combined ^
  --model runs\detect\assets\training_runs\yolov8s_accessibility\weights\best.pt ^
  --output assets\models\yolov8s_accessibility.pt ^
  --project assets\training_runs ^
  --name yolov8s_accessibility_full ^
  --epochs 80 ^
  --imgsz 640 ^
  --batch 8 ^
  --device 0 ^
  --patience 20 ^
  --workers 4 ^
  > assets\training_runs\yolov8s_accessibility_full.out.log 2> assets\training_runs\yolov8s_accessibility_full.err.log
