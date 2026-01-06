"""
Memory system: lưu trữ lịch sử trò chuyện, quản lý context cho AI
"""
import json
import os
from datetime import datetime
from pathlib import Path


class ConversationMemory:
    """Quản lý lịch sử trò chuyện"""
    
    def __init__(self, memory_file: str = "memory.json", max_history: int = 20):
        """
        Args:
            memory_file: đường dẫn file lưu memory
            max_history: số lượng tin nhắn tối đa lưu trữ
        """
        self.memory_file = memory_file
        self.max_history = max_history
        self.history = []
        self.load_memory()
    
    def load_memory(self):
        """Tải lịch sử trò chuyện từ file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                print(f"✅ Loaded {len(self.history)} messages from memory")
            except Exception as e:
                print(f"⚠ Error loading memory: {e}")
                self.history = []
        else:
            print("📝 New conversation started")
            self.history = []
    
    def save_memory(self):
        """Lưu lịch sử trò chuyện vào file"""
        try:
            data = {"history": self.history, "last_updated": datetime.now().isoformat()}
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Error saving memory: {e}")
    
    def add_message(self, role: str, content: str):
        """Thêm tin nhắn vào lịch sử"""
        message = {
            "role": role,  # "user" hoặc "assistant"
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(message)
        
        # Giữ lại chỉ max_history tin nhắn gần nhất
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.save_memory()
    
    def get_context(self, include_timestamps: bool = False) -> list:
        """Lấy context cho LLM (định dạng cho API)"""
        context = []
        for msg in self.history:
            if include_timestamps:
                context.append({
                    "role": msg["role"],
                    "content": f"[{msg['timestamp']}] {msg['content']}"
                })
            else:
                context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        return context
    
    def clear_memory(self):
        """Xóa toàn bộ lịch sử"""
        self.history = []
        self.save_memory()
        print("🗑 Memory cleared")
    
    def print_history(self, limit: int = 10):
        """In lịch sử trò chuyện gần đây"""
        print("\n📜 Conversation History:")
        print("=" * 60)
        
        for msg in self.history[-limit:]:
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            timestamp = msg.get("timestamp", "")[:19]  # YY-MM-DD HH:MM:SS
            print(f"\n{role} [{timestamp}]:")
            print(f"  {msg['content'][:100]}...")  # In 100 ký tự đầu
        
        print("\n" + "=" * 60)
    
    def get_stats(self) -> dict:
        """Lấy thống kê cuộc trò chuyện"""
        user_msgs = len([m for m in self.history if m["role"] == "user"])
        assistant_msgs = len([m for m in self.history if m["role"] == "assistant"])
        
        return {
            "total_messages": len(self.history),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "memory_file": self.memory_file
        }


# # Test
# if __name__ == "__main__":
#     memory = ConversationMemory("test_memory.json", max_history=10)
    
#     # Thêm tin nhắn
#     memory.add_message("user", "Xin chào, bạn tên là gì?")
#     memory.add_message("assistant", "Tôi là một trợ lý AI. Tôi có thể giúp bạn với nhiều tác vụ.")
#     memory.add_message("user", "Hôm nay là ngày mấy?")
#     memory.add_message("assistant", "Hôm nay là ngày 6 tháng 1 năm 2026.")
    
#     # In lịch sử
#     memory.print_history()
    
#     # Lấy stats
#     stats = memory.get_stats()
#     print(f"\n📊 Stats: {stats}")
    
#     # Lấy context cho LLM
#     context = memory.get_context()
#     print(f"\n💬 Context cho LLM: {len(context)} messages")
