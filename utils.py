from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

vector_db_path = "vectorstores/db_faiss"
def initialize_retrievers():
    try:
        # Tải và xử lý PDF
        loader = PyPDFLoader("data/working.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""],
            length_function=len,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        
        # Tạo BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(all_splits)
        bm25_retriever.k = 5
        
        # Tạo FAISS retriever
        embedding_model = GPT4AllEmbeddings(model_file=r"models\vinallama-7b-chat_q5_0.gguf")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        
        return bm25_retriever, db
    except Exception as e:
        print(f"Lỗi khi khởi tạo retrievers: {str(e)}")
        return None, None
    

def hybrid_retriever(query, bm25_retriever, vector_store):
    # Lấy kết quả từ cả BM25 và FAISS
    bm25_docs = bm25_retriever.get_relevant_documents(query)[:5]
    faiss_docs = vector_store.similarity_search(query, k=5)
    
    # Kết hợp và loại bỏ trùng lặp
    seen_content = set()
    final_docs = []
    
    # Xen kẽ kết quả từ cả hai nguồn
    for i in range(max(len(bm25_docs), len(faiss_docs))):
        if i < len(bm25_docs):
            if bm25_docs[i].page_content not in seen_content:
                seen_content.add(bm25_docs[i].page_content)
                final_docs.append(bm25_docs[i])
                
        if i < len(faiss_docs):
            if faiss_docs[i].page_content not in seen_content:
                seen_content.add(faiss_docs[i].page_content)
                final_docs.append(faiss_docs[i])

    return final_docs[:5]

