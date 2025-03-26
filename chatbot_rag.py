from utils import initialize_retrievers
from smart_ans import smart_response

# Khởi tạo retrievers một lần
bm25_retriever, vector_store = None, None

def initialize():
    global bm25_retriever, vector_store
    bm25_retriever, vector_store = initialize_retrievers()

def rag_search(query):
    try:
        global bm25_retriever, vector_store
        if bm25_retriever is None or vector_store is None:
            initialize()
        response = smart_response(query, bm25_retriever, vector_store)
        return response
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        return "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."

if __name__ == "__main__":
    try:
        initialize()  # Khởi tạo ngay từ đầu
        while True:
            try:
                query = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
                if query.lower() == "exit":
                    print("👋 Tạm biệt!")
                    break
                
                response = rag_search(query)
                if response:
                    print("\nCâu trả lời:", response)
                else:
                    print("\n❌ Không tìm được câu trả lời phù hợp.")
                    
            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {str(e)}")
                print("Vui lòng thử lại.")
    finally:
        # Giải phóng tài nguyên
        if vector_store:
            vector_store = None
        if bm25_retriever:
            bm25_retriever = None
