import sys
import os

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


from qanything_kernel.qanything_server.handler_search import *
# from qanything_kernel.core.local_doc_search_cpu import LocalDocSearch
from qanything_kernel.core.local_doc_search_npu import LocalDocSearch
from sanic import Sanic
from sanic import response as sanic_response
import argparse
from sanic.worker.manager import WorkerManager

WorkerManager.THRESHOLD = 6000

# 接收外部参数mode
parser = argparse.ArgumentParser()
# mode必须是local或online
parser.add_argument('--mode', type=str, default='local', help='local or online')
parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu or npu')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--device_id', type=str, default='0', help='device id')

# parser.add_argument('--rerank_model_path', type=str, help='rerank model path')
# parser.add_argument('--embedding_model_path', type=str, help='embedding model path')
# 检查是否是local或online，不是则报错
args = parser.parse_args()
if args.mode not in ['local', 'online']:
    raise ValueError('mode must be local or online')



app = Sanic("QAnything")
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
async def init_local_doc_search(app, loop):
    local_doc_search = LocalDocSearch()
    local_doc_search.init_cfg(args)
    print(f'init local_doc_search in {args.mode}', flush=True)
    app.ctx.local_doc_search = local_doc_search



# app.add_route(upload_weblink, "/api/qanything/upload_weblink", methods=['POST'])  # tags=["上传网页链接"]
# app.add_route(get_total_status, "/api/qanything/get_total_status", methods=['POST'])  # tags=["获取所有知识库状态"]
# app.add_route(clean_files_by_status, "/api/qanything/clean_files_by_status", methods=['POST'])  # tags=["清理数据库"]
# app.add_route(delete_docs, "/api/qanything/delete_files", methods=['POST'])  # tags=["删除文件"] 
# app.add_route(rename_knowledge_base, "/api/qanything/rename_knowledge_base", methods=['POST'])  # tags=["重命名知识库"] 
# app.add_route(upload_faqs, "/api/qanything/upload_faqs", methods=['POST'])  # tags=["上传FAQ"]
# app.add_route(get_qa_info, "/api/qanything/get_qa_info", methods=['POST'])  # tags=["获取QA信息"]


app.add_route(document, "/api/docs", methods=['GET'])
app.add_route(list_kbs, "/api/qanything/list_knowledge_base", methods=['POST'])  # tags=["知识库列表"] 
app.add_route(list_docs, "/api/qanything/list_files", methods=['POST'])  # tags=["文件列表"]
app.add_route(new_knowledge_base, "/api/qanything/new_knowledge_base", methods=['POST'])  # tags=["新建知识库"]
app.add_route(delete_knowledge_base, "/api/qanything/delete_knowledge_base", methods=['POST'])  # tags=["删除知识库"]
app.add_route(document_parser, "/api/qanything/document_parser", methods=['POST'])  # tags=["解析文件"]
app.add_route(document_parser_embedding, "/api/qanything/document_parser_embedding", methods=['POST'])  # tags=["解析文件并保存"]
app.add_route(chunk_embedding, "/api/qanything/chunk_embedding", methods=['POST'])  # tags=["切片数据保存"]
app.add_route(question_rag_search, "/api/qanything/question_rag_search", methods=['POST'])  # tags=["问答接口"]
app.add_route(get_files_statu, "/api/qanything/get_files_statu", methods=['POST'])  # tags=["获取指定文件状态"]


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8777, workers=10, access_log=False)
