from typing import Dict, List, Tuple
from datetime import datetime
import requests
import pytz

class PatternManager:
    def __init__(self):
        self.chitchat_patterns: Dict[str, str] = {
            'greeting': r'(xin chào|hello|hi|chào|hey)',
            'family': r'(gia đình|nhà|bao nhiêu người|mấy người)',
            'name': r'(tên|tên gì|là ai|bạn là)',
            'time': r'(mấy giờ|thời gian|giờ giấc)',
            'date': r'(ngày|thứ|hôm nay)',
            'health': r'(khỏe|thế nào)',
            'age': r'(mấy tuổi|bao nhiêu tuổi)',
            'goodbye': r'(tạm biệt|bye|goodbye|gặp lại sau|hẹn gặp lại)',
            'thanks': r'(cảm ơn|thank|cảm ơn bạn rất nhiều)',
            'weather': r'(thời tiết|nắng|mưa)',
            'general': r'(ừ|ok|được|vâng|yes)',
            'sad': r'(buồn|cảm thấy buồn|cảm thấy buồn bã|cảm thấy buồn phiền|thất tình)',
            'emotion': r'(vui|buồn|giận|thích|yêu|ghét|nhớ)',
            'opinion': r'(nghĩ|suy nghĩ|ý kiến|quan điểm)',
            'personal': r'(sở thích|thích gì|đam mê|yêu thích)',
            'ability': r'(có thể|làm được|khả năng|biết)',
            'confirmation': r'(phải không|đúng không|có phải|có đúng)',
            'smalltalk': r'(thế à|vậy à|ừ nhỉ|thật không|thật à)',
            'chat_fuck': r'(con cặc|địt mẹ mày|fuck you|ngu|điên)'
        }
        
        self.doc_patterns: Dict[str, str] = {
            'what': r'(là gì|như thế nào|ra sao|thế nào|gì|những gì|định nghĩa|khái niệm)',
            'how': r'(làm sao|như thế nào|cách|phương pháp|bằng cách nào|quy trình|các bước)',
            'why': r'(tại sao|vì sao|lý do|nguyên nhân|do đâu)',
            'when': r'(khi nào|lúc nào)',
            'where': r'(ở đâu|chỗ nào|nơi nào)',
            'which': r'(loại nào|cái nào|nào)',
            'explain': r'(giải thích|phân tích|trình bày)',
            'compare': r'(so sánh|khác nhau)',
            'list': r'(liệt kê|kể|nêu)',
            'define': r'(định nghĩa|khái niệm|nghĩa)',
            'find': r'(tìm|tra cứu|xem|kiếm|lookup)',
            'analyze': r'(phân tích|đánh giá|nhận xét|review)',
            'summarize': r'(tóm tắt|tổng hợp|summary|overview)',
            'technical': r'(kỹ thuật|công nghệ|phương pháp|algorithm|thuật toán)',
            'reference': r'(tham khảo|nguồn|trích dẫn|cite)',
            
        }
        
        self.responses: Dict[str, List[str]] = {
            'greeting': [
                "Xin chào! Tôi là AI HÀN BẢO, tôi có thể giúp gì cho bạn?",
                "Chào bạn! Rất vui được gặp bạn.",
            ],
            'family': [
                "Tôi là một AI nên không có gia đình theo nghĩa thông thường. Nhưng tôi có một cộng đồng người dùng và nhà phát triển rất tuyệt vời!",
                "Là một AI, tôi không có gia đình như con người. Tôi được tạo ra để hỗ trợ mọi người như bạn.",
                "Tôi là trợ lý ảo nên không có gia đình thực sự. Nhưng tôi rất vui được trò chuyện với bạn!"
            ],
            'general': [
                "Vâng, tôi hiểu rồi. Bạn cần giúp đỡ gì không?",
                "OK, tôi đang lắng nghe bạn.",
                "Được, bạn cứ nói tiếp nhé."
            ],
            'thanks': [
                "Không có gì, đó là nhiệm vụ của tôi!",
                "Rất vui khi được giúp bạn!",
                "Không có chi bạn nhé!"
            ],
            'name': [
                "Tôi là AI HÀN BẢO, tôi có thể giúp gì cho bạn?",
            ],
            'age': [
                "Tôi là AI HÀN BẢO, tôi không có tuổi, còn bạn mấy tuổi?",
            ],
            'health': [
                "Tôi rất khỏe, còn bạn thì sao?",
            ],
            'weather': [
                "hỏi chị google nhá",
            ],
            'goodbye': [
                "Tạm biệt bạn! Hẹn gặp lại sau nhé!",
            ],
            'thanks': [
                "Không có gì, đó là nhiệm vụ của tôi!",
                "Rất vui khi được giúp bạn!",
                "Không có chi bạn nhé!"
            ],
            'sad': ["Đừng buồn nhé, tôi sẽ luôn ở đây để hỗ trợ bạn",
                    "Tôi rất vui khi được trò chuyện với bạn",
                    "Tôi không biết nói gì để làm bạn vui hơn, nhưng tôi sẽ luôn ở đây để hỗ trợ bạn",],
            'chat_fuck': ["ê ê không chửi thề nha mày, đcm mày",]
        }

    def add_chitchat_pattern(self, intent: str, pattern: str) -> None:
        """Thêm pattern mới cho chitchat"""
        self.chitchat_patterns[intent] = pattern

    def add_doc_pattern(self, intent: str, pattern: str) -> None:
        """Thêm pattern mới cho document"""
        self.doc_patterns[intent] = pattern

    def add_response(self, intent: str, responses: List[str]) -> None:
        """Thêm câu trả lời cho một intent"""
        self.responses[intent] = responses

    def get_all_patterns(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Lấy tất cả các pattern"""
        return self.chitchat_patterns, self.doc_patterns

    def get_current_time(self) -> str:
        """Lấy thời gian hiện tại theo múi giờ Việt Nam"""
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vn_tz)
        return current_time.strftime("%H:%M:%S")
        
    def get_current_date(self) -> str:
        """Lấy ngày và thứ hiện tại bằng tiếng Việt"""
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        current_date = datetime.now(vn_tz)
        weekday_names = {
            0: 'Thứ Hai',
            1: 'Thứ Ba',
            2: 'Thứ Tư',
            3: 'Thứ Năm',
            4: 'Thứ Sáu',
            5: 'Thứ Bảy',
            6: 'Chủ Nhật'
        }
        weekday = weekday_names[current_date.weekday()]
        return f"{weekday}, ngày {current_date.strftime('%d/%m/%Y')}"


    def get_responses(self, text: str) -> List[str]:
        """Cập nhật phương thức get_responses"""
        import re
        text = text.lower()
        
        # Kiểm tra trước nếu là câu hỏi liên quan đến tài liệu
        for intent, pattern in self.doc_patterns.items():
            if re.search(pattern, text):
                return ["DOCUMENT_QUERY", intent]
        
        # Nếu không phải câu hỏi tài liệu, kiểm tra các intent chitchat
        found_intents = []
        for intent, pattern in self.chitchat_patterns.items():
            if re.search(pattern, text):
                found_intents.append(intent)
                
        # Xử lý theo thứ tự ưu tiên cho chitchat
        if 'time' in found_intents:
            current_time = self.get_current_time()
            if current_time is None:
                return ["Xin lỗi, không thể lấy thời gian hiện tại."]
            return [f"Bây giờ là {current_time}"]
            
        if 'date' in found_intents:
            return [self.get_current_date()]
            
        if 'weather' in found_intents:
            return self.responses['weather']
            
        # Xử lý các intent chitchat khác
        for intent in found_intents:
            if intent not in ['time', 'date', 'weather']:
                return self.responses.get(intent, ["Xin lỗi, tôi chưa có câu trả lời cho nội dung này."])
        
        return ["GENERAL_QUERY"]  # Nếu không match với pattern nào
