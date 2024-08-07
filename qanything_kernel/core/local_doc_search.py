from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, \
    OCR_MODEL_PATH, SEARCH_EXPAND_CONTENT, SEARCH_EXPAND_CONTENT_LENGTH, ADD_FILENAME_TO_EMBEDDING
from typing import List
import os
import time
from langchain.schema import Document
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.connector.database.faiss.faiss_client import FaissClient
from qanything_kernel.connector.rerank.rerank_backend import RerankBackend
from qanything_kernel.connector.embedding.embedding_backend import EmbeddingBackend
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.core.tools.web_search_tool import duckduckgo_search
from qanything_kernel.dependent_server.ocr_server.ocr import OCRQAnything
from qanything_kernel.utils.general_utils import num_tokens
from .local_file import LocalFile
import traceback
import base64
import numpy as np
import requests

class LocalDocSearch:
    def __init__(self):
        self.embeddings: EmbeddingBackend = None
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_size: int = CHUNK_SIZE
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.faiss_client: FaissClient = None
        self.mysql_client: KnowledgeBaseManager = None
        self.local_rerank_backend: RerankBackend = None
        self.ocr_reader: OCRQAnything = None
        self.mode: str = None
        self.model: str = None
        self.use_paddleocr = False
        self.ocr_url = 'http://127.0.0.1:8010/ocr'

    def get_ocr_result(self, input: dict):
        if self.use_paddleocr:
            response = requests.post(self.ocr_url, json=input, timeout=60)
            response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
            result = response.json()['results']
            if result[0] is not None:
                result = [i[1][0] for line in result for i in line]
            else:
                result = []
            debug_logger.info(f"paddle ocr result: {result}")
            return result
        else:
            img_file = input['img64']
            height = input['height']
            width = input['width']
            channels = input['channels']
            binary_data = base64.b64decode(img_file)
            img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
            ocr_res = self.ocr_reader(img_array)
            if len(ocr_res)>0:
                ocr_res = [line for line in ocr_res if line]
            else:
                ocr_res = []
            res = [line for line in ocr_res if line]
            debug_logger.info(f"local ocr result: {res}")
            return res

    def init_cfg(self, args):
        self.rerank_top_k = 3
        self.device = args.device
        self.use_paddleocr = args.use_paddleocr
        if not self.device=="cpu":
            self.device = self.device+":"+str(args.device_id)
        
        if args.backend == "onnx":
            debug_logger.info(f"init onnx backend")
            from qanything_kernel.connector.rerank.rerank_onnx_backend import RerankOnnxBackend
            from qanything_kernel.connector.embedding.embedding_onnx_backend import EmbeddingOnnxBackend
            self.local_rerank_backend: RerankOnnxBackend = RerankOnnxBackend(device=self.device)
            self.embeddings: EmbeddingOnnxBackend = EmbeddingOnnxBackend(device=self.device)     
        elif args.backend == "torch":
            debug_logger.info(f"init torch backend")
            from qanything_kernel.connector.rerank.rerank_torch_backend import RerankTorchBackend
            from qanything_kernel.connector.embedding.embedding_torch_backend import EmbeddingTorchBackend
            self.local_rerank_backend: RerankTorchBackend = RerankTorchBackend(device=self.device)
            self.embeddings: EmbeddingTorchBackend = EmbeddingTorchBackend(device=self.device)
        else:
            debug_logger.info(f"error input backend: {args.backend}")
            raise ValueError(f"error input backend: {args.backend}")
        
        self.mysql_client = KnowledgeBaseManager()
        if not self.use_paddleocr:
            self.ocr_reader = OCRQAnything(model_dir=OCR_MODEL_PATH, device=self.device)  # 省显存
            debug_logger.info(f"OCR DEVICE: {self.ocr_reader.device}")
        else:
            debug_logger.info(f"use paddleocr server api")
        self.faiss_client = FaissClient(self.mysql_client, self.embeddings)

    async def insert_files_to_faiss(self, user_id, kb_id, local_files: List[LocalFile]):
        debug_logger.info(f'insert_files_to_faiss: {kb_id}')
        success_list = []
        failed_list = []

        for local_file in local_files:
            start = time.time()
            try:
                debug_logger.info(f'start split {local_file.file_name}')
                local_file.split_file_to_docs(self.get_ocr_result)
                
                # 新增将split后的docs保存到本地文档中，便于可以查看切片后的文档内容
                local_file.save_docs_to_local()

                if local_file.file_name.endswith('.faq'):
                    content_length = len(local_file.docs[0].metadata['faq_dict']['question']) + len(local_file.docs[0].metadata['faq_dict']['answer'])
                else:
                    content_length = sum([len(doc.page_content) for doc in local_file.docs])
            
                end = time.time()
                self.mysql_client.update_content_length(local_file.file_id, content_length)
                debug_logger.info(f'split time: {end - start} {len(local_file.docs)}')
                self.mysql_client.update_chunk_size(local_file.file_id, len(local_file.docs))
                add_ids = await self.faiss_client.add_document(local_file.docs)
                insert_time = time.time()
                debug_logger.info(f'insert time: {insert_time - end}')
                self.mysql_client.update_file_status(local_file.file_id, status='green')
                success_list.append(local_file)
            except Exception as e:
                error_info = f'split error: {traceback.format_exc()}'
                debug_logger.error(error_info)
                self.mysql_client.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
        debug_logger.info(
            f"insert_to_faiss: success num: {len(success_list)}, failed num: {len(failed_list)}")
        
        # 更新faiss缓存文件状态
        self.faiss_client.delete_faiss_cache(kb_id)

    def deduplicate_documents(self, source_docs):
        unique_docs = set()
        deduplicated_docs = []
        for doc in source_docs:
            if doc.page_content not in unique_docs:
                unique_docs.add(doc.page_content)
                doc.metadata['faiss_search_score'] = float(doc.metadata['score'])  #add faiss search score
                deduplicated_docs.append(doc)
        return deduplicated_docs
    
    def add_filename_to_docs(self, source_docs):
        for doc in source_docs:
            doc.page_content = f"<<{doc.metadata['file_name']}>>:\n{doc.page_content}"
        return source_docs

    def del_filename_in_docs(self, source_docs):
        for doc in source_docs:
            doc.page_content = doc.page_content.replace(f"<<{doc.metadata['file_name']}>>:\n", "")
        return source_docs

    
    def expand_page_by_context(self, doc, context_length=600, positions=[-1,1]):
        # debug_logger.info(f"\n\n\nexpand_page_by_context doc: {doc}\n\n\n")
        if ADD_FILENAME_TO_EMBEDDING:
            file_name_tmp = doc.metadata['file_name']
            file_name_tmp = "<<"+os.path.splitext(file_name_tmp)[0]+">>:\n"
            doc.page_content = doc.page_content.replace(file_name_tmp, '')
        
        new_positions = []
        for position in positions:
            position_doc = self.faiss_client.get_neighbors_documents(doc, position)
            if ADD_FILENAME_TO_EMBEDDING and position_doc:
                file_name_tmp = position_doc.metadata['file_name']
                file_name_tmp = "<<"+os.path.splitext(file_name_tmp)[0]+">>:\n"
                position_doc.page_content = position_doc.page_content.replace(file_name_tmp, '')
            
            if position<0:
                if position_doc:
                    doc.page_content = position_doc.page_content+"\n"+doc.page_content
                    new_positions.append(position-1)
            elif position>0:
                if position_doc:
                    doc.page_content = doc.page_content+"\n"+position_doc.page_content
                    new_positions.append(position+1)
            else:
                debug_logger.error(f"position error: {position}")
                return doc
        
        if len(doc.page_content) >= context_length or len(new_positions)==0:
            if ADD_FILENAME_TO_EMBEDDING:
                doc.page_content = "<<"+os.path.splitext(doc.metadata['file_name'])[0]+">>:\n"+doc.page_content
            # debug_logger.info(f"\n\nreturn expend doc: {doc}\n\n")
            return doc
        else:
            return self.expand_page_by_context(doc, context_length=context_length, positions=new_positions)
    
    
    async def local_doc_search(self, query, kb_ids, score_threshold=0.35, rerank: bool = True, merge: bool = True):
        source_documents = await self.get_source_documents(query, kb_ids, merge=merge)
        deduplicated_docs = self.deduplicate_documents(source_documents)
        
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        debug_logger.info(f"retrieval docs num: {len(retrieval_documents)}")
        # debug_logger.info(f"retrieval docs: {retrieval_documents}")


        # 对检索文档进行扩展
        if SEARCH_EXPAND_CONTENT:

            # for index in range(len(retrieval_documents)):
            #     if len(retrieval_documents[index].page_content)< SEARCH_EXPAND_CONTENT_LENGTH:
            #         debug_logger.info(f"before expand page by context: {len(retrieval_documents[index].page_content)}")
            #         retrieval_documents[index] = copy.deepcopy(self.expand_page_by_context(retrieval_documents[index], context_length=SEARCH_EXPAND_CONTENT_LENGTH, positions=[1]))
            #         debug_logger.info(f"after expand page by context: {len(retrieval_documents[index].page_content)}")
            #         debug_logger.info(f"after expand page by context: {retrieval_documents[index].page_content}")
            # debug_logger.info(f"\n\n\n expand retrieval docs: {retrieval_documents}")

            expand_retrieval_documents=[]

            for doc in retrieval_documents:
                if len(doc.page_content)< SEARCH_EXPAND_CONTENT_LENGTH:
                    # debug_logger.info(f"\n\nbefore expand page by context: {len(doc.page_content)}")
                    # debug_logger.info(f"before expand page by context: {doc}")
                    # doc = self.expand_page_by_context(doc, context_length=SEARCH_EXPAND_CONTENT_LENGTH, positions=[1])
                    if ADD_FILENAME_TO_EMBEDDING:
                        file_name_tmp = doc.metadata['file_name']
                        file_name_tmp = "<<"+os.path.splitext(file_name_tmp)[0]+">>:\n"
                        doc.page_content = doc.page_content.replace(file_name_tmp, '')
                    
                    # 往后扩充长度
                    position = 1
                    while len(doc.page_content)<SEARCH_EXPAND_CONTENT_LENGTH:
                        position_doc = self.faiss_client.get_neighbors_documents(doc, position)
                        if ADD_FILENAME_TO_EMBEDDING and position_doc:
                            file_name_tmp = "<<"+os.path.splitext(position_doc.metadata['file_name'])[0]+">>:\n"
                            position_doc.page_content = position_doc.page_content.replace(file_name_tmp, '')
                        
                        if position<0 and position_doc:
                            doc.page_content = position_doc.page_content+"\n"+doc.page_content
                        if position>0 and position_doc:
                            doc.page_content = doc.page_content+"\n"+position_doc.page_content
                        if position_doc is None:
                            # debug_logger.warn(f"position error: {position}")
                            break
                        position += 1
                    
                    if ADD_FILENAME_TO_EMBEDDING:
                        doc.page_content = file_name_tmp+doc.page_content

                    # debug_logger.info(f"\n\nafter expand page by context: {len(doc.page_content)}")
                    # debug_logger.info(f"after expand page by context: {doc}")
                
                expand_retrieval_documents.append(doc)
            retrieval_documents = expand_retrieval_documents
            # debug_logger.info(f"\n\n\n expand retrieval docs: {retrieval_documents}")
        

        if len(retrieval_documents) > 1 and rerank:
            debug_logger.info(f"use rerank, rerank docs num: {len(retrieval_documents)}")
            # rerank需要的query必须是改写后的, 不然会丢一些信息
            retrieval_documents = self.rerank_documents(query, retrieval_documents)
        # 删除掉分数低于阈值的文档
        if score_threshold:
            tmp_documents = [item for item in retrieval_documents if float(item.metadata['score']) > score_threshold]
            if tmp_documents:
                retrieval_documents = tmp_documents
        
        retrieval_documents = retrieval_documents[: self.rerank_top_k]
        # debug_logger.info(f"\n\n\n rerank top{self.rerank_top_k} retrieval docs: {retrieval_documents}\n\n\n ")
        return retrieval_documents
        
        

    def get_web_search(self, queries, top_k=None):
        if not top_k:
            top_k = self.top_k
        query = queries[0]
        web_content, web_documents = duckduckgo_search(query)
        source_documents = []
        for doc in web_documents:
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            source_documents.append(doc)
        return web_content, source_documents



    def web_page_search(self, query, top_k=None):
        # 防止get_web_search调用失败，需要try catch
        try:
            web_content, source_documents = self.get_web_search([query], top_k)
        except Exception as e:
            debug_logger.error(f"web search error: {e}")
            return []

        return source_documents


    async def get_source_documents(self, query, kb_ids, cosine_thresh=None, top_k=None, merge: bool = True):
        if not top_k:
            top_k = self.top_k
        source_documents = []
        t1 = time.time()
        filter = lambda metadata: metadata['kb_id'] in kb_ids
        # filter = None
        debug_logger.info(f"query: {query}")
        docs = await self.faiss_client.search(kb_ids, query, filter=filter, top_k=top_k, merge=merge)
        debug_logger.info(f"query_docs: {len(docs)}")
        t2 = time.time()
        debug_logger.info(f"faiss search time: {t2 - t1}")
        for idx, doc in enumerate(docs):
            if doc.metadata['file_name'].endswith('.faq'):
                faq_dict = doc.metadata['faq_dict']
                doc.page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                nos_keys = faq_dict.get('nos_keys')
                doc.metadata['nos_keys'] = nos_keys
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            # doc.metadata['embed_version'] = self.embeddings.getModelVersion
            source_documents.append(doc)
        if cosine_thresh:
            source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents


    def generate_prompt(self, query, source_docs, prompt_template):
        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def rerank_documents(self, query, source_documents):
        if num_tokens(query) > 300:  # tokens数量超过300时不使用local rerank
            return source_documents

        scores = self.local_rerank_backend.predict(query, [doc.page_content for doc in source_documents])
        debug_logger.info(f"rerank scores: {scores}")
        for idx, score in enumerate(scores):
                source_documents[idx].metadata['score'] = score
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)
        return source_documents

    async def retrieve(self, query, kb_ids, need_web_search=False, rerank: bool = False, merge: bool = True):
        retrieval_documents = await self.local_doc_search(query, kb_ids, rerank=rerank, merge=merge)
        if need_web_search:
            retrieval_documents.extend(self.web_page_search(query, top_k=3))
            debug_logger.info(f"add web_search retrieval_documents: {retrieval_documents}")
            retrieval_documents = self.rerank_documents(query, retrieval_documents)
            debug_logger.info(f"add web_search reranked retrieval_documents: {retrieval_documents}")
        return retrieval_documents

    async def get_knowledge_based_answer(self, query, kb_ids,
                                         rerank: bool = False,
                                         merge: bool = True):
        
        #retrieval_queries = [query]
        retrieval_documents = await self.retrieve(query, kb_ids, need_web_search=False, rerank=rerank, merge=merge)

        return retrieval_documents
        
        


