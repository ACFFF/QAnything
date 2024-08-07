#!/bin/bash

# cd /workspace/qanything_local || exit  #部署
update_or_append_to_env() {
  local key=$1
  local value=$2
  local env_file="qanything_kernel/configs/model_config.py"

  # 检查键是否存在于配置文件中
  if grep -q "^${key}=" "$env_file"; then
    # 如果键存在，则更新它的值
    sed -i "/^${key}=/c\\${key}=${value}" "$env_file"
  else
    # 如果键不存在，则追加键值对到文件
    echo "${key}=${value}" >> "$env_file"
  fi
}


# 初始化参数
system="Linux"
milvus_port=19530
qanything_port=8777
device="gpu"
device_id=0
workers=1
use_paddleocr=true

# 使用getopts解析命令行参数
while getopts ":p:d:i:w:" opt; do
  case $opt in
    p) qanything_port="$OPTARG"
    ;;
    d) device="$OPTARG"
    ;;
    i) device_id="$OPTARG"
    ;;
    w) workers="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


# 确保必需参数已提供
if [ -z "$qanything_port" ]; then
    echo "必须提供 --qanything_port 参数。"
    exit 1
fi

echo "device: $device"
echo "device_id: $device_id"
echo "qanything_port: $qanything_port"
echo "workers: $workers"


if [ "$use_paddleocr" == "true" ]; then
    echo "use paddle ocr"
    if [ "$device" == "gpu" ]; then
        echo "Using GPU for PaddleOCR"
        export OCR_USE_GPU=True
    fi
    nohup python3 -u qanything_kernel/dependent_server/ocr_serve/ocr_server.py > ./logs/debug_logs/ocr_server.log 2>&1 &
    echo "The ocr service is ready!"
    echo "OCR服务已就绪!"
    
    python3 -m qanything_kernel.qanything_server.sanic_api_search --host 0.0.0.0 --port $qanything_port \
    --device $device --device_id $device_id --workers $workers --use_paddleocr 
else
    echo "use local ocr"
    python3 -m qanything_kernel.qanything_server.sanic_api_search --host 0.0.0.0 --port $qanything_port \
    --device $device --device_id $device_id --workers $workers
fi

sleep 1
# 启动qanything-server服务

    


# echo -e "即将启动Milvus服务"
# echo "运行Milvus的命令是："
# echo "CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.milvus_server.milvus_server --host 0.0.0.0 --port $milvus_port"

# sleep 1
# # 启动Milvus服务
