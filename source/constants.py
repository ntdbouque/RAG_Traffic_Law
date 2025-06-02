from config.config import get_config

cfg = get_config("/workspace/competitions/Sly/Duy_NCKH_2025/config/config.yaml")

# Generation model configuration
STREAM = cfg.MODEL.STREAM
SERVICE = cfg.MODEL.SERVICE
TEMPERATURE = cfg.MODEL.TEMPERATURE
MODEL_ID = cfg.MODEL.MODEL_ID

QA_PROMPT = (
    "Dưới đây là một nội dung của một Điều/hoặc những Khoản nhỏ của một điều trích từ một thông tư/nghị định/luật: \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Dựa vào thông tin trên, hãy trả lời câu hỏi sau: {query_str}\n\
    Lưu ý luôn luôn suy nghĩ kĩ trước khi trả lời. Để trả lời câu hỏi vế đầu tiên bạn phải trích dẫn lại ý chính của câu hỏi và trả lời thẳng vào câu hỏi.\
    Sau đó trích dẫn lại nội dung bạn đã tham chiếu kèm theo vị trí của đoạn tham chiếu đó trong Luật. Nếu không biết hãy trả lời thành thật và không đưa ra thông tin sai lệch"
)

# Contextual RAG configuration
EMBEDDING_MODEL = cfg.CONTEXTUAL_RAG.EMBEDDING_MODEL
CONTEXTUAL_SERVICE = cfg.CONTEXTUAL_RAG.SERVICE

CONTEXTUAL_PROMPT = """\
Dưới đây là thông tin bổ sung về ngữ cảnh của chunk này:

- **Tên nghị định**: {DECREE_NAME}
- **Tên chương**: {CHAPTER_NAME}
- **Vị trí của khoản trong điều**: {SARTICLE_POSITION}
- **Toàn bộ nội dung của điều chứa chunk**: {ARTICLE_TITLE}:\n{FULL_ARTICLE_CONTENT}

---

### Chunk đầu vào:
{CHUNK_CONTENT}

---

### Yêu cầu:
Hãy viết **một câu ngắn gọn** nhằm mô tả ngữ cảnh tổng quát của chunk, giúp định vị nội dung trong toàn bộ tài liệu nhằm cải thiện khả năng tìm kiếm.

**Yêu cầu cụ thể:**
- Không lặp lại nguyên văn nội dung chunk.
- Không cần nêu rõ số điều, khoản, nghị định.
- Không liệt kê các chi tiết cụ thể hay hành vi rời rạc.
- Phải làm rõ **đối tượng áp dụng** của quy định (ví dụ xe thì là xe gì, người thì đó là cá nhân, cơ quan hay tổ chức nào).
- Câu mô tả cần mang tính tổng quát, đủ rõ ràng để người đọc hình dung được chủ đề chính.

**Chỉ trả lời đúng 1 câu mô tả nội dung chính của chunk, và trả lời dưới dạng JSON với cấu trúc sau:**

```json
{{"context": "Nội dung câu mô tả chính của chunk"}}
"""

METADATA_PROMPT = '''
Từ đoạn văn bản dưới đây, hãy trích xuất các thông tin sau một cách rõ ràng và chính xác:

1. Tiêu đề: tiêu đề của bộ luật (ví dụ Luật Dân Sự)
2. Số của luật: Số hiệu hoặc mã số của bộ luật  
3. Ngày tháng năm ban hành: Ngày, tháng, năm mà luật được ban hành hoặc thông qua (định dạng DD/MM/YY).  
4. Địa điểm ban hành: Địa điểm hoặc cơ quan ban hành luật (nếu có thông tin).  

Nếu không tìm thấy một trong các thông tin trên, hãy ghi rõ 'Không có thông tin'.

Văn bản:
'{context_str}'

Định dạng đầu ra:
- Tiêu đề: [Tiêu đề]
- Luật số: [Số hiệu]
- Ngày tháng năm ban hành: [Ngày tháng năm]
- Địa điểm ban hành: [Địa điểm]
'''

CONTEXTUAL_MODEL = cfg.CONTEXTUAL_RAG.MODEL

CONTEXTUAL_RAG_COLLECTION_NAME = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME

QDRANT_URL = cfg.CONTEXTUAL_RAG.QDRANT_URL
ELASTIC_SEARCH_URL = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_URL
ELASTIC_SEARCH_INDEX_NAME = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME

SEMANTIC_WEIGHT = cfg.CONTEXTUAL_RAG.SEMANTIC_WEIGHT
BM25_WEIGHT = cfg.CONTEXTUAL_RAG.BM25_WEIGHT
TOP_N = cfg.CONTEXTUAL_RAG.TOP_N

SUPPORTED_FILE_EXTENSIONS = [".pdf"]

# Agent configuration
AGENT_TYPE = cfg.AGENT.TYPE