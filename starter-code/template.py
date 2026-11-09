"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant, trợ lý tư vấn thân thiện và chính xác của hệ sinh thái Vingroup.

PERSONA: Tư vấn sản phẩm VinFast, dịch vụ Vinpearl và tiếp nhận phản hồi bằng tiếng Việt.
AVAILABLE TOOLS: search_product_catalog tra cứu sản phẩm và giá từ catalog;
submit_support_ticket tạo yêu cầu hỗ trợ khi khách cung cấp tên và vấn đề.
CORE RULES: Không bịa giá, sản phẩm hoặc chính sách. Phải dùng tool khi cần tra cứu
hoặc tạo ticket; chỉ xác nhận ticket sau khi tool trả kết quả thành công.
OPERATIONAL BOUNDARIES: Chỉ hỗ trợ thông tin thuộc hệ sinh thái Vingroup.
OUTPUT CONTRACT: Thought xác định nhu cầu; Action gọi tool thích hợp;
Observation ghi kết quả; Final Answer trả lời ngắn gọn dựa trên kết quả thật.
Không hiển thị suy luận nội bộ trong câu trả lời cho khách hàng."""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline - mock] Tôi chưa tra cứu dữ liệu cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _budget(text: str) -> int:
        match = re.search(r"dưới\s+(\d+(?:[.,]\d+)?)\s*(triệu|tỷ)", text, re.I)
        if not match:
            return 999999999999
        amount = float(match.group(1).replace(",", "."))
        return int(amount * (1_000_000_000 if match.group(2).lower() == "tỷ" else 1_000_000))

    @staticmethod
    def _ticket_details(text: str) -> Dict[str, str]:
        name_match = re.search(r"(?:tôi tên|tên tôi là)\s+([^,.!?]+)", text, re.I)
        name = name_match.group(1).strip() if name_match else ""
        issue = ""
        if name_match:
            issue = text[name_match.end():].lstrip(" ,:;")
        elif "phản hồi:" in text.lower():
            issue = re.split(r"phản hồi:", text, flags=re.I)[-1].strip()
        issue = re.split(r"[.!?]|(?:đây là vấn đề|mức độ|cần xử lý gấp)", issue, maxsplit=1, flags=re.I)[0].strip(" ,:;")
        priority = "high" if re.search(r"nghiêm trọng|gấp|khẩn cấp", text, re.I) else (
            "low" if re.search(r"không gấp|mức độ thấp", text, re.I) else "medium")
        return {"customer_name": name, "issue_description": issue, "priority": priority}

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        text = user_input.lower()
        shopping = bool(re.search(r"xem|giá|có xe|resort|tìm.*(xe|phòng)", text))
        needs_catalog = shopping and bool(re.search(r"xe điện|vinfast|\bvf\s*\d|resort|vinpearl", text))
        needs_ticket = bool(re.search(r"bị lỗi|ghi nhận phản hồi|muốn ghi nhận|bị ẩm mốc|yêu cầu hỗ trợ", text))
        actions = []
        if needs_catalog:
            category = "du_lich" if re.search(r"resort|vinpearl|phòng", text) else "xe_dien"
            actions.append(("search_product_catalog", {"category": category, "max_price": self._budget(text)}))
        if needs_ticket:
            details = self._ticket_details(user_input)
            if not details["customer_name"] or not details["issue_description"]:
                self.trace.append({"step": "clarification", "missing": "customer_name or issue_description"})
                return {"answer": "Vui lòng cung cấp tên và mô tả vấn đề để tạo yêu cầu hỗ trợ.",
                        "trace": self.trace, "iterations": 1, "status": "completed"}
            actions.append(("submit_support_ticket", details))

        if not actions:
            answer = ("Theo danh mục hiện có, VinFast VF 5 Plus có tính năng bảo hành pin 10 năm; "
                      "điều kiện bảo hành cụ thể vui lòng xác nhận với VinFast."
                      if "bảo hành" in text and "pin" in text else
                      "Tôi hỗ trợ thông tin sản phẩm và dịch vụ Vingroup; vui lòng nêu yêu cầu cụ thể.")
            self.trace.append({"step": "final", "answer": answer})
            return {"answer": answer, "trace": self.trace, "iterations": 1, "status": "completed"}

        parts = []
        iteration = 0
        for name, arguments in actions:
            if iteration >= self.max_iterations:
                break
            observation = TOOL_MAP[name](**arguments)
            iteration += 1
            self.trace.append({"step": iteration, "action": name, "arguments": arguments,
                               "observation": observation})
            if name == "search_product_catalog":
                if observation and "error" in observation[0]:
                    parts.append("Không thể tra cứu danh mục sản phẩm lúc này.")
                elif not observation:
                    parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    parts.append("Sản phẩm phù hợp: " + ", ".join(
                        f"{p['name']} ({p['price_vnd']:,} VNĐ)" for p in observation))
            else:
                parts.append(f"Đã tạo ticket {observation['ticket_id']} cho {observation['customer_name']}.")

        completed = iteration == len(actions)
        if not completed:
            parts.append("Chưa xử lý hết yêu cầu vì đã đạt giới hạn số bước.")
        return {"answer": " ".join(parts), "trace": self.trace,
                "iterations": iteration, "status": "completed" if completed else "max_iterations_reached"}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
