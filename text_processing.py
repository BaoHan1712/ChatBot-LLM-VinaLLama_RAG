from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = GPT4AllEmbeddings(model_file="models/vinallama-7b-chat_q5_0.gguf")
def load_pdf_data():
    pdf_path = "data/baocao.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Điều chỉnh kích thước chunk và overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Tạo và lưu vector store với metric tốt hơn
    vector_store = FAISS.from_documents(
        all_splits, 
        embeddings,
        distance_strategy="METRIC_INNER_PRODUCT"
    )
    vector_store.save_local("faiss_index")

def test_search():
    # Test tìm kiếm
    query = "hãy cho tôi biết kỹ thuạt Phát hiện biên cạnh (Edge Detection)"
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query, k=3)
    for doc in docs:
        print(doc.page_content)


# load_pdf_data()
test_search()

