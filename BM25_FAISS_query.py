import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


embeddings = GPT4AllEmbeddings(model_file="models/vinallama-7b-chat_q5_0.gguf")

def load_pdf_data():
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
    
    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 5 

    vector_store = FAISS.from_documents(
        all_splits, 
        embeddings,
        distance_strategy="METRIC_INNER_PRODUCT"
    )
    vector_store.save_local("faiss_index")

    return bm25_retriever

def print_results(results, query):
    print(f"\n🔍 Top 5 kết quả phù hợp nhất cho câu hỏi: '{query}'")
    
    for i, doc in enumerate(results[:5], 1):
        print(f"\n{i}. {doc.page_content}")
        print("-" * 50)
        
    if not results:
        print("❌ Không tìm thấy kết quả phù hợp.")

def hybrid_search(query, bm25_retriever, vector_store):
    """
    Hybrid search sử dụng cả BM25 và FAISS để tìm kiếm kết quả tốt nhất.
    """
    # Lấy kết quả từ cả hai phương pháp
    bm25_docs = bm25_retriever.get_relevant_documents(query)[:5]  
    faiss_docs = vector_store.similarity_search(query, k=5)  
    
    # Kết hợp kết quả theo tỷ lệ 50-50
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

def preprocess_query(query):
    query = query.lower().strip()
    
    # Loại bỏ các từ stop word tiếng Việt phổ biến
    stop_words = {"là", "và", "các", "của", "có", "những", "để", "trong", "cho"}
    query_words = [word for word in query.split() if word not in stop_words]
    
    # Tạo các biến thể của query
    variations = [
        " ".join(query_words),  # Query gốc đã loại stop words
        " ".join(query_words[:3]),  # 3 từ đầu tiên
        " ".join(query_words[-3:])  # 3 từ cuối cùng
    ]
    
    return variations

def enhanced_test_search(bm25_retriever, vector_store):
    while True:
        query = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ").strip()
        if query.lower() == "exit":
            print("👋 Tạm biệt!")
            break
            
        query_variations = preprocess_query(query)
        
        all_results = []
        for q in query_variations:
            results = hybrid_search(q, bm25_retriever, vector_store)
            all_results.extend(results)
        
        unique_results = list({doc.page_content: doc for doc in all_results}.values())
        print_results(unique_results[:5], query)

if __name__ == "__main__":
    if not os.path.exists("faiss_index"):
        bm25_retriever = load_pdf_data()
    else:
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
        bm25_retriever = BM25Retriever.from_documents(all_splits)
        bm25_retriever.k = 5

    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    enhanced_test_search(bm25_retriever, vector_store)
