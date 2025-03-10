from utils import initialize_retrievers
from smart_ans import smart_response

def rag_search(query):
    bm25_retriever, vector_store = initialize_retrievers()
    response = smart_response(query, bm25_retriever, vector_store)
    return response

if __name__ == "__main__":
    while True:
        query = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
        if query.lower() == "exit":
            print("👋 Tạm biệt!")
            break
        response = rag_search(query)
        print("\nCâu trả lời:", response)
