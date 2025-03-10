from datetime import datetime
import re
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from pattern_manager import PatternManager
from langchain.chains import LLMChain
from utils import hybrid_retriever

pattern_manager = PatternManager()


def preprocess_query(query: str) -> str:
    """Tiền xử lý và chuẩn hóa câu hỏi"""
    # Chuẩn hóa câu hỏi
    query = query.lower().strip()
    
    # Loại bỏ các ký tự đặc biệt
    special_chars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in special_chars:
        query = query.replace(char, ' ')
    
    # Loại bỏ khoảng trắng thừa
    query = ' '.join(query.split())
    
    return query

def classify_query(query: str) -> tuple[str, float, str]:
    """Phân loại và cho điểm câu hỏi thông minh hơn"""
    query = preprocess_query(query)
    pm = PatternManager()
    
    # Tính điểm với trọng số cho chitchat
    chitchat_score = 0
    chitchat_weights = {
        'greeting': 1.5,
        'emotion': 1.3,
        'personal': 1.2,
        'smalltalk': 1.1,
        'opinion': 0.9
    }
    
    found_chitchat_intent = None
    for intent, pattern in pm.chitchat_patterns.items():
        if re.search(pattern, query):
            weight = chitchat_weights.get(intent, 1.0)
            chitchat_score += weight
            if not found_chitchat_intent:
                found_chitchat_intent = intent
            
    # Tính điểm với trọng số cho document
    doc_score = 0
    doc_weights = {
        'technical': 1.5,
        'analyze': 1.3,
        'find': 1.2,
        'what': 1.1,
        'how': 1.1
    }
    
    found_doc_intent = None
    for intent, pattern in pm.doc_patterns.items():
        if re.search(pattern, query):
            weight = doc_weights.get(intent, 1.0)
            doc_score += weight
            if not found_doc_intent:
                found_doc_intent = intent
    
    # Quyết định loại câu hỏi với ngưỡng điểm
    if chitchat_score > doc_score and chitchat_score >= 1.0:
        return "chitchat", chitchat_score, found_chitchat_intent
    elif doc_score >= 0.8:
        return "document", doc_score, found_doc_intent
    else:
        # Nếu không rõ ràng, chuyển xuống LLM xử lý
        return "general", 0.5, "general"

def smart_response(query: str, bm25_retriever, vector_store):
    """Xử lý câu hỏi và trả về câu trả lời phù hợp"""
    # Phân loại câu hỏi trước
    query_type, confidence, intent = classify_query(query)
    
    # Kiểm tra trong PatternManager cho chitchat
    pattern_manager = PatternManager()
    responses = pattern_manager.get_responses(query)
    
    # Nếu có câu trả lời chitchat rõ ràng
    if responses[0] not in ["DOCUMENT_QUERY", "GENERAL_QUERY"]:
        return responses[0]

    # Khởi tạo LLM
    llm = CTransformers(
        model="models/vinallama-7b-chat_q5_0.gguf",
        model_type="llama",
        temperature=0.2,
        max_new_tokens=128,
        top_k=30,
        top_p=0.9
    )

    # Xử lý câu hỏi liên quan đến tài liệu
    if responses[0] == "DOCUMENT_QUERY" or (query_type == "document" and confidence >= 0.8):
        # Truy vấn RAG
        docs = hybrid_retriever(query, bm25_retriever, vector_store)
        if not docs:
            return "Tôi không tìm thấy thông tin liên quan trong tài liệu."
            
        context = "\n".join([doc.page_content for doc in docs])
        
        doc_prompt = PromptTemplate(
            template="""Bạn là AI HÀN BẢO. Hãy sử dụng thông tin sau đây để trả lời câu hỏi một cách ngắn gọn và chính xác:

            Context: {context}
            
            Question: {question}
            
            Yêu cầu:
            1. Chỉ trả lời dựa trên thông tin trong context
            2. Trả lời ngắn gọn, súc tích
            3. Không lặp lại câu hỏi trong câu trả lời
            4. Nếu không tìm thấy thông tin, hãy nói: "Tôi không tìm thấy thông tin về điều này trong tài liệu."
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        chain = LLMChain(llm=llm, prompt=doc_prompt)
        response = chain.run(context=context, question=query)
        
        # Loại bỏ phần lặp lại câu hỏi nếu có
        if response.lower().startswith(query.lower()):
            response = response[len(query):].strip(" ,.:")
            
        return response
    
    # Xử lý câu hỏi chitchat
    elif query_type == "chitchat" and confidence >= 0.7:
        chitchat_prompt = PromptTemplate(
            template="""Bạn là AI HÀN BẢO, một trợ lý AI thân thiện. 
            Hãy trả lời câu hỏi sau một cách tự nhiên và thân thiện:
            
            Câu hỏi: {question}
            
            Intent: {intent}
            Confidence: {confidence}
            
            Trả lời:""",
            input_variables=["question", "intent", "confidence"]
        )
        
        chain = LLMChain(llm=llm, prompt=chitchat_prompt)
        response = chain.run(question=query, intent=intent, confidence=confidence)
        return f"[Chitchat - {intent}] {response}"
    
    # Xử lý câu hỏi thông thường
    else:
        general_prompt = PromptTemplate(
            template="""Bạn là AI HÀN BẢO. Hãy trả lời câu hỏi sau một cách thân thiện và vui nhộn:
            
            Câu hỏi: {question}
            
            Intent: {intent}
            Confidence: {confidence}
            
            Trả lời:""",
            input_variables=["question", "intent", "confidence"]
        )
        
        chain = LLMChain(llm=llm, prompt=general_prompt)
        return chain.run(question=query, intent=intent, confidence=confidence)
