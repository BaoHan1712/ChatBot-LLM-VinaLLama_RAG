from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = LlamaCpp(
        model_path=model_file,
        temperature=0.2,
        max_tokens=512,
        n_ctx=2048,
        top_p=0.9,
        verbose=False,
        n_gpu_layers=32,
        n_batch=512       
    )
    return llm

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

# Tao simple chain
def create_qa_chain(prompt, llm, bm25_retriever, vector_store):
    def get_context(query):
        docs = hybrid_search(query, bm25_retriever, vector_store)
        return "\n".join([doc.page_content for doc in docs])
        
    def qa_chain(query):
        context = get_context(query)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(context=context, question=query)
        return {"result": response}
        
    return qa_chain

# Read tu VectorDB
def read_vectors_db():
    # Đọc PDF và tạo chunks
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

def hybrid_search(query, bm25_retriever, vector_store):
    # Lấy kết quả từ cả hai phương pháp
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

def main():
    # Khởi tạo các thành phần
    bm25_retriever, vector_store = read_vectors_db()
    llm = load_llm(model_file)
    
    # Tạo prompt template
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi, trả lời ngắn gọn. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = creat_prompt(template)
    
    # Tạo chain với hybrid search
    llm_chain = create_qa_chain(prompt, llm, bm25_retriever, vector_store)
    
    # Vòng lặp hỏi đáp
    while True:
        try:
            question = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
            if question.lower() == 'exit':
                print("Tạm biệt!")
                break
                
            response = llm_chain(question)
            print("\nCâu trả lời:", response['result'])
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\nCó lỗi xảy ra: {str(e)}")
            print("Vui lòng thử lại.")

if __name__ == "__main__":
    main()