#!/bin/bash

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
use_cpu=false


# 使用getopts解析命令行参数
while getopts ":q:c:w:" opt; do
  case $opt in
    q) qanything_port="$OPTARG"
    ;;
    c) use_cpu=true
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



if [ "$use_cpu" = true ]; then
    use_cpu_option="--use_cpu"
else
    use_cpu_option=""
fi


echo -e "即将启动后端服务"
echo "运行qanything-server的命令是："
echo "CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.qanything_server.sanic_api_search --host 0.0.0.0 --port $qanything_port --model_size  ${workers:+--workers "$workers"}"

sleep 1
# 启动qanything-server服务
CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.qanything_server.sanic_api_search --host 0.0.0.0 --port $qanything_port \
    $use_cpu_option \
    ${workers:+--workers "$workers"}

# nohup CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.qanything_server.sanic_api_liaoning --host 0.0.0.0 --port $qanything_port \
#     $use_cpu_option \
#     ${workers:+--workers "$workers"} > test0608.log 2>&1 &


# echo -e "即将启动Milvus服务"
# echo "运行Milvus的命令是："
# echo "CUDA_VISIBLE_DEVICES=0 python3 -m qanything_kernel.milvus_server.milvus_server --host 0.0.0.0 --port $milvus_port"

# sleep 1
# # 启动Milvus服务
