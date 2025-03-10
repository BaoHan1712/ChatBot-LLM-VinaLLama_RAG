from langchain_community.embeddings import GPT4AllEmbeddings
import numpy as np

embeddings = GPT4AllEmbeddings(model_file="models/vinallama-7b-chat_q5_0.gguf")

# Chuyển list thành numpy array để có thể sử dụng .shape
embedding_vector = np.array(embeddings.embed_query("test"))
print("Embedding dimension:", len(embedding_vector))
