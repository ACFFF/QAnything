from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, FAISS_LOCATION, FAISS_CACHE_SIZE, ADD_FILENAME_TO_EMBEDDING, \
    VECTOR_SEARCH_SCORE_THRESHOLD
from typing import Optional, Union, Callable, Dict, Any, List, Tuple
from langchain_community.vectorstores.faiss import dependable_faiss_import
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.utils.general_utils import num_tokens
from functools import lru_cache
import shutil
import stat
import os
import platform

os_system = platform.system()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因


class SelfInMemoryDocstore(InMemoryDocstore):
    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in memory dictionary.

        Args:
            texts: dictionary of id -> document.

        Returns:
            None
        """
        # overlapping = set(texts).intersection(self._dict)
        # if overlapping:
        #     raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        # self._dict = {**self._dict, **texts}
        self._dict.update(texts)


# @lru_cache(FAISS_CACHE_SIZE)
# def load_vector_store(faiss_index_path, embeddings):
#     debug_logger.info(f'load faiss index: {faiss_index_path}')
#     return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


# 存到内存中，减少读取时间
from collections import OrderedDict
faiss_cache_obj=OrderedDict()
def load_vector_store(faiss_index_path, embeddings):
    if faiss_index_path in faiss_cache_obj.keys():
        debug_logger.info(f'load faiss index in cache: {faiss_index_path}')
        faiss_cache_obj.move_to_end(faiss_index_path)
        return faiss_cache_obj[faiss_index_path]

    debug_logger.info(f'load faiss index: {faiss_index_path}')
    faiss_cache_obj[faiss_index_path] = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


    if len(faiss_cache_obj) > FAISS_CACHE_SIZE:
        faiss_index_path_to_remove = faiss_cache_obj.popitem(last=False)[0]
        debug_logger.info(f'remove faiss index from cache: {faiss_index_path_to_remove}')
    return faiss_cache_obj[faiss_index_path]


class FaissClient:
    def __init__(self, mysql_client: KnowledgeBaseManager, embeddings):
        self.mysql_client: KnowledgeBaseManager = mysql_client
        self.embeddings = embeddings
        self.faiss_client: FAISS = None
        self.kb_ids: List[str] = []
        self.save_faiss_status = False

    def _load_kb_to_memory(self, kb_ids):
        debug_logger.info(f'_load_kb_to_memory kb_ids: {kb_ids}')
        self.kb_ids = kb_ids
        self.faiss_client = None
        for kb_id in kb_ids:
            faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
            if os.path.exists(faiss_index_path):
                faiss_client: FAISS = load_vector_store(faiss_index_path, self.embeddings)
            else:
                faiss = dependable_faiss_import()
                index = faiss.IndexFlatL2(768)
                docstore = SelfInMemoryDocstore()
                debug_logger.info(f'init FAISS kb_id: {kb_id}')
                faiss_client: FAISS = FAISS(self.embeddings, index, docstore, index_to_docstore_id={})
            if self.faiss_client is None:
                self.faiss_client = faiss_client
            else:
                try:
                    self.faiss_client.merge_from(faiss_client)
                    debug_logger.info(f'merge FAISS kb_id: {kb_id}')
                except ValueError:
                    raise ValueError(f'遗留数据与新版本不匹配，请删除{os.path.dirname(FAISS_LOCATION)}文件夹（清空所有知识库）后重新启动服务并重新创建知识库')
        debug_logger.info(f'FAISS load kb_ids: {kb_ids}')

    async def search(self, kb_ids, query, filter: Optional[Union[Callable, Dict[str, Any]]] = None,
                     top_k=VECTOR_SEARCH_TOP_K, merge: bool = True):
        if self.faiss_client is None or self.kb_ids != kb_ids:
            self._load_kb_to_memory(kb_ids)
        # filter = {'page': 1}
        if filter is None:
            filter = {}
        debug_logger.info(f'FAISS search: {query}, {filter}, {top_k}')
        docs_with_score = await self.faiss_client.asimilarity_search_with_score(query, k=top_k, filter=filter,
                                                                                fetch_k=200,
                                                                                score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD)
        debug_logger.info(f'FAISS search result number: {len(docs_with_score)}')
        for doc, score in docs_with_score:
            doc.metadata['score'] = score
        docs = [doc for doc, score in docs_with_score]
        # docs_with_score = self.merge_docs(docs)
        # return docs_with_score

        import copy
        docs_deepcopy = copy.deepcopy(docs)
        if merge:
            debug_logger.info("use merge docs")
            docs_deepcopy = self.merge_docs(docs_deepcopy)
        return docs_deepcopy

    def merge_docs(self, docs):
        # 把docs按照file_id进行合并，但是需要对所有file_id相同的doc根据chunk_id先排序，chunk_id相邻的doc合并
        # 合并如果添加了标题，需要进行删除
        merged_docs = []
        docs = sorted(docs, key=lambda x: (x.metadata['file_id'], x.metadata['chunk_id']))
        for doc in docs:
            if not merged_docs or merged_docs[-1].metadata['file_id'] != doc.metadata['file_id']:
                merged_docs.append(doc)
                file_name = "<<"+os.path.splitext(merged_docs[-1].metadata["file_name"])[0]+">>:\n"
                if ADD_FILENAME_TO_EMBEDDING and file_name not in merged_docs[-1].page_content:
                    merged_docs[-1].page_content = file_name + merged_docs[-1].page_content
                    debug_logger.warn(f"Manual add file name in doc page content!{doc.page_content[:50]}{doc.metadata}")
            else:
                if merged_docs[-1].metadata['chunk_id'] == doc.metadata['chunk_id'] - 1:
                    if num_tokens(merged_docs[-1].page_content + doc.page_content) <= 800:
                        if ADD_FILENAME_TO_EMBEDDING:
                            file_name_tmp = doc.metadata['file_name']
                            file_name_tmp = "<<"+os.path.splitext(file_name_tmp)[0]+">>:\n"
                            if not doc.page_content.startswith(file_name_tmp):
                                debug_logger.error(f'doc not start with file name: {doc.page_content[:50]}{doc.metadata}')
                            doc.page_content = doc.page_content.replace(file_name_tmp, '')
                        debug_logger.info(f'Merge doc: {merged_docs[-1].metadata["file_name"]} chunkid: {merged_docs[-1].metadata["chunk_id"]} and {doc.metadata["chunk_id"]}')
                        merged_docs[-1].page_content += '\n' + doc.page_content
                        merged_docs[-1].metadata['chunk_id'] = doc.metadata['chunk_id'] #注意，chunk_id以后面的为准，则merged后扩充只能往后扩充，往前则会重复
                        
                    else:
                        # print('NOT MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                        merged_docs.append(doc)
                        file_name = "<<"+os.path.splitext(merged_docs[-1].metadata["file_name"])[0]+">>:\n"
                        if ADD_FILENAME_TO_EMBEDDING and file_name not in merged_docs[-1].page_content:
                            merged_docs[-1].page_content = file_name + merged_docs[-1].page_content
                            debug_logger.warn(f"Manual add file name in doc page content!{doc.page_content[:50]}{doc.metadata}")
                else:
                    # print('NOT MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                    merged_docs.append(doc)
                    file_name = "<<"+os.path.splitext(merged_docs[-1].metadata["file_name"])[0]+">>:\n"
                    if ADD_FILENAME_TO_EMBEDDING and file_name not in merged_docs[-1].page_content:
                        merged_docs[-1].page_content = file_name + merged_docs[-1].page_content
                        debug_logger.warn(f"Manual add file name in doc page content!{doc.page_content[:50]}{doc.metadata}")
            

        return merged_docs

    async def add_document(self, docs):
        # 判断是否有文档正在保存，如果有则等待其完成后再保存添加
        while self.save_faiss_status:
            import asyncio
            await asyncio.sleep(1)
        self.save_faiss_status = True
        
        kb_id = docs[0].metadata['kb_id']
        if self.faiss_client is None or self.kb_ids != [kb_id]:
            self._load_kb_to_memory([kb_id])
        add_ids = await self.faiss_client.aadd_documents(docs)
        # doc带上id存入Document表中
        chunk_id = 0
        for doc, add_id in zip(docs, add_ids):
            self.mysql_client.add_document(add_id, chunk_id, doc.metadata['file_id'], doc.metadata['file_name'],
                                           doc.metadata['kb_id'])
            chunk_id += 1
        debug_logger.info(f'add documents number: {len(add_ids)}')
        faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
        self.faiss_client.save_local(faiss_index_path)
        debug_logger.info(f'save faiss index: {faiss_index_path}')
        os.chmod(os.path.dirname(faiss_index_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        
        self.save_faiss_status = False
        return add_ids

    def delete_documents(self, kb_id, file_ids=None):
        if self.faiss_client is None or self.kb_ids != [kb_id]:
            self._load_kb_to_memory([kb_id])
        doc_ids = []
        if file_ids is None:
            kb_index_path = os.path.join(FAISS_LOCATION, kb_id)
            if os.path.exists(kb_index_path):
                shutil.rmtree(kb_index_path)
                self.faiss_client = None
                debug_logger.info(f'delete kb_id: {kb_id}, {kb_index_path}')
                return
        elif file_ids:
            doc_ids = self.mysql_client.get_documents_by_file_ids(file_ids)
        doc_ids = [doc_id[0] for doc_id in doc_ids]
        if not doc_ids:
            debug_logger.info(f'no documents to delete')
            return
        try:
            res = self.faiss_client.delete(doc_ids)
            debug_logger.info(f'delete documents: {res}')
            faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
            self.faiss_client.save_local(faiss_index_path)
            debug_logger.info(f'save faiss index: {faiss_index_path}')
            os.chmod(os.path.dirname(faiss_index_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except ValueError as e:
            debug_logger.warning(f'delete documents not find docs')
   

    
    def get_neighbors_documents(self, doc, position=0):
        # debug_logger.info(f'get neighbors documents: {doc.metadata}')        
        if self.faiss_client is None or self.kb_ids != [doc.metadata['kb_id']]:
            self._load_kb_to_memory([doc.metadata['kb_id']])
            
        # 根据doc的file_id, 在mysql_client中查看文件的信息
        position_doc_info = self.mysql_client.get_documents_by_fileid_chunkid(doc.metadata['file_id'], doc.metadata['chunk_id']+position)
        position_doc = None
        if position_doc_info:
            position_doc_docstore_id = position_doc_info[0][0]
            position_doc = self.faiss_client.docstore.search(position_doc_docstore_id)
        # debug_logger.info(f"position_doc_info: {position_doc_info}")
        # debug_logger.info(f"position_doc:{position_doc}")
        
        import copy
        position_doc_deepcopy = copy.deepcopy(position_doc)
        return position_doc_deepcopy
        
