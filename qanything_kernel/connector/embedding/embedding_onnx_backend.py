import numpy as np
import time
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel 
from qanything_kernel.configs.model_config import LOCAL_EMBED_MODEL_PATH
from qanything_kernel.utils.custom_log import debug_logger
from qanything_kernel.connector.embedding.embedding_backend import EmbeddingBackend


class EmbeddingOnnxBackend(EmbeddingBackend):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=="cpu")
        self.device = device
        self.return_tensors = "np"
        sess_options = SessionOptions()
        sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        if "cpu" in self.device:
            providers = ['CPUExecutionProvider']
        elif "gpu" in self.device:
            device_id = self.device.split(":")[-1]
            if device_id.isdigit():
                device_id = int(device_id)
            else:
                device_id = 0
            providers = [
                (
                    'CUDAExecutionProvider',
                    {
                        "device_id": device_id
                    }
                 )
            ]
        elif "npu" in self.device:
            device_id = self.device.split(":")[-1]
            if device_id.isdigit():
                device_id = int(device_id)
            else:
                device_id = 0
            providers = [
                (
                    "CANNExecutionProvider",
                    {
                        "device_id": device_id,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "npu_mem_limit": 20 * 1024 * 1024 * 1024,
                        "enable_cann_graph": True,
                    },
                )
            ]
        self._session = InferenceSession(LOCAL_EMBED_MODEL_PATH, sess_options=sess_options, providers=providers)
        debug_logger.info(f"EmbeddingClient: model_path: {LOCAL_EMBED_MODEL_PATH}")
        debug_logger.info(f"EmbeddingOnnxBackend initialized with device: {self.device}")

    def get_embedding(self, sentences, max_length):
        inputs_onnx = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors=self.return_tensors)
        inputs_onnx = {k: v for k, v in inputs_onnx.items()}
        start_time = time.time()
        outputs_onnx = self._session.run(output_names=['output'], input_feed=inputs_onnx)
        debug_logger.info(f"onnx infer time: {time.time() - start_time}")
        embedding = outputs_onnx[0][:,0]
        debug_logger.info(f'embedding shape: {embedding.shape}')
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr

        return embeddings_normalized.tolist()
