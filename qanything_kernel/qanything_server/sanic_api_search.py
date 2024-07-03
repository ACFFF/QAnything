import os
import sys
import time

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取当前脚本的父目录的路径，即`qanything_server`目录
current_dir = os.path.dirname(current_script_path)

# 获取`qanything_server`目录的父目录，即`qanything_kernel`
parent_dir = os.path.dirname(current_dir)

# 获取根目录：`qanything_kernel`的父目录
root_dir = os.path.dirname(parent_dir)

# 将项目根目录添加到sys.path
sys.path.append(root_dir)



# 读取输入参数
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--host', dest='host', default='0.0.0.0', help='set host for qanything server')
parser.add_argument('--port', dest='port', default=8777, type=int, help='set port for qanything server')
parser.add_argument('--workers', dest='workers', default=4, type=int, help='sanic server workers number')
parser.add_argument('--device', dest='device', default='npu', help='显卡设备，可以设置为npu, gpu, cpu')
parser.add_argument('--device_id', dest='device_id', default='0', help='cuda device id for qanything server')
parser.add_argument('--offline', dest='offline', default=False, help='offline mode')
args = parser.parse_args()

# 针对离线推理环境，需要设置tiktoken以及unstructured库部分组件联网检查的问题
if args.offline:
    # 设置tiktoken联网检查为False
    tiktoken_cache_dir = "/workspace/qanything_kernel/model/tiktoken_model"
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    
    # 设置unstructured联网检查为False
    os.environ["SCARF_NO_ANALYTICS"] = True
    os.environ["DO_NOT_TRACK"] = True

# 控制分词器（Tokenizers）在处理文本时的并行性
os.environ["TOKENIZERS_PARALLELISM"] = "false"


    


from sanic import Sanic
from sanic import response as sanic_response
from sanic.worker.manager import WorkerManager
from .handler_search import *
from qanything_kernel.core.local_doc_search import LocalDocSearch
from qanything_kernel.utils.custom_log import debug_logger

WorkerManager.THRESHOLD = 6000

app = Sanic("RAG-LIAONING")
# 设置请求体最大为 400MB
app.config.REQUEST_MAX_SIZE = 400 * 1024 * 1024

# 将 /qanything 路径映射到 ./dist/qanything 文件夹，并指定路由名称
# app.static('/qanything/', 'qanything_kernel/qanything_server/dist/qanything/', name='qanything', index="index.html")

# CORS中间件，用于在每个响应中添加必要的头信息
@app.middleware("response")
async def add_cors_headers(request, response):
    # response.headers["Access-Control-Allow-Origin"] = "http://10.234.10.144:5052"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"  # 如果需要的话


@app.middleware("request")
async def handle_options_request(request):
    if request.method == "OPTIONS":
        headers = {
            # "Access-Control-Allow-Origin": "http://10.234.10.144:5052",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Credentials": "true"  # 如果需要的话
        }
        return sanic_response.text("", headers=headers)


@app.before_server_start
async def init_local_doc_qa(app, loop):
    start = time.time()
    local_doc_qa = LocalDocSearch()
    local_doc_qa.init_cfg(args=args)
    debug_logger.info(f"LocalDocSearch started in {time.time() - start} seconds.")
    app.ctx.local_doc_qa = local_doc_qa

@app.after_server_start
async def print_info(app, loop):
    print("已启动后端服务。", flush=True)



app.add_route(document, "/api/docs", methods=['GET'])   # tags=["接口文档"]
app.add_route(new_knowledge_base, "/api/qanything/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
app.add_route(delete_knowledge_base, "/api/qanything/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"]
app.add_route(document_parser, "/api/qanything/document_parser", methods=['POST'])  # tags=["解析文件"]
app.add_route(document_parser_embedding, "/api/qanything/document_parser_embedding", methods=['POST'])  # tags=["解析文件并保存"]
app.add_route(chunk_embedding, "/api/qanything/chunk_embedding", methods=['POST'])  # tags=["切片数据保存"]
app.add_route(question_rag_search, "/api/qanything/question_rag_search", methods=['POST'])  # tags=["问答接口"]
app.add_route(list_kbs, "/api/qanything/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
app.add_route(list_docs, "/api/qanything/list_files", methods=['POST'])  # tags=["文件列表"]
app.add_route(get_files_statu, "/api/qanything/get_files_statu", methods=['POST'])  # tags=["获取指定文件状态"]
app.add_route(upload_faqs, "/api/qanything/upload_faqs", methods=['POST'])  # tags=["上传FAQ"]



if __name__ == "__main__":
    
    try:
        # 尝试以指定的workers数量启动应用
        app.run(host=args.host, port=args.port, workers=args.workers, access_log=False)
    except Exception as e:
        debug_logger.info(f"启动多worker模式失败: {e}，尝试以单进程模式启动。")
        # 如果出现异常，则退回到单进程模式
        app.run(host=args.host, port=args.port, single_process=True, access_log=False)
    
    # 由于有用户启动时上下文环境报错，使用单进程模式：
    # app.run(host=args.host, port=args.port, single_process=True, access_log=False)




