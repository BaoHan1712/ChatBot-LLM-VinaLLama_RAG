from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

def initialize_retrievers():
    # Tải và xử lý PDF
    pdf_path = "data/working.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", ",", " ", ""],
        length_function=len,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Khởi tạo BM25
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 3
    
    # Tải FAISS
    embeddings = GPT4AllEmbeddings(model_file="models/vinallama-7b-chat_q5_0.gguf")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    return bm25_retriever, vector_store

def hybrid_retriever(query, bm25_retriever, vector_store):
    # Lấy kết quả từ cả BM25 và FAISS
    bm25_docs = bm25_retriever.get_relevant_documents(query)[:3]
    faiss_docs = vector_store.similarity_search(query, k=3)
    
    # Kết hợp và loại bỏ trùng lặp
    seen_content = set()
    final_docs = []
    
    for doc in bm25_docs + faiss_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            final_docs.append(doc)
    
    return final_docs[:3] 