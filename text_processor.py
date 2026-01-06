"""
Text processor: xử lý ký hiệu đặc biệt, giới hạn độ dài, lọc nhiễu
"""
import re
import unicodedata


def clean_special_chars(text: str) -> str:
    """Xóa hoặc escape các ký hiệu đặc biệt như **, __, ##, etc."""
    if not text:
        return ""
    
    # Remove markdown formatting completely
    text = text.replace("**", "")       # ** → xóa
    text = text.replace("__", "")       # __ → xóa
    text = text.replace("##", "")       # ## → xóa
    text = text.replace("```", "")      # ``` → xóa
    text = text.replace("`", "")        # ` → xóa
    text = text.replace("*", "")        # * → xóa
    text = text.replace("_", "")        # _ → xóa
    text = text.replace("#", "")        # # → xóa
    
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    
    # Remove excessive special chars: !!!!, ????, ----
    text = re.sub(r"!{2,}", "!", text)  # !! → !
    text = re.sub(r"\?{2,}", "?", text) # ?? → ?
    text = re.sub(r"-{3,}", "--", text) # --- → --
    text = re.sub(r"={3,}", "==", text) # === → ==
    
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def limit_length(text: str, max_chars: int = 500) -> str:
    """Giới hạn độ dài text, cắt ở dấu câu gần nhất"""
    if not text or len(text) <= max_chars:
        return text
    
    # Cắt text
    truncated = text[:max_chars].strip()
    
    # Tìm dấu kết thúc gần nhất (. ! ? ; :)
    for char in [".  ", "!  ", "?  ", ";  ", ":  "]:
        pos = truncated.rfind(char)
        if pos > max_chars * 0.7:  # Ít nhất 70% độ dài
            return truncated[:pos + 1].strip()
    
    # Fallback: cắt ở khoảng trắng cuối cùng
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].strip() + "..."
    
    return truncated + "..."


def process_ai_response(text: str, max_chars: int = 500) -> str:
    """Xử lý đầy đủ: lọc ký hiệu + giới hạn độ dài"""
    text = clean_special_chars(text)
    text = limit_length(text, max_chars)
    return text


def sanitize_user_input(text: str) -> str:
    """Sanitize user input: xóa control chars, collapse whitespace"""
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    
    # Remove control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# # Test
# if __name__ == "__main__":
#     # Test clean_special_chars
#     test1 = "Xin chào! **Đây là thử nghiệm** với __, ##tags###"
#     print(f"Clean: {clean_special_chars(test1)}")
    
#     # Test limit_length
#     test2 = "Đây là một câu rất dài. Nó chứa nhiều thông tin. Chúng tôi cần cắt nó xuống còn 500 ký tự. Đây là phần thứ hai. Và đây là phần thứ ba nữa."
#     print(f"Limited: {limit_length(test2, 100)}")
    
#     # Test full process
#     test3 = "**Hello** world!!!?? `code` here ___underline___"
#     result = process_ai_response(test3, 50)
#     print(f"Processed: {result}")
