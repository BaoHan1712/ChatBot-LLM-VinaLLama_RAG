# from langchain_community.embeddings import LlamaCppEmbeddings
# import numpy as np

# embeddings = LlamaCppEmbeddings(
#     model_path="models/vinallama-7b-chat_q5_0.gguf",
#     n_ctx=2048,
#     n_gpu_layers=32,
#     n_batch=512,
#     verbose=False
# )

# # Test ansering người dùng
# embedding_vector = np.array(embeddings.embed_query("test"))
# print("Embedding dimension:", len(embedding_vector))

from transformers import file_utils
print(file_utils.default_cache_path)
