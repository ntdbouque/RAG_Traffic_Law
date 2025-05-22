from config.config import get_config

cfg = get_config("config/config.yaml")

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

CONTEXTUAL_PROMPT = """<document>
{WHOLE_DOCUMENT}
</document>
Đây là đoạn nội dung mà chúng ta muốn đặt trong ngữ cảnh của toàn bộ tài liệu
Bên dưới đây là {ARTICLE_TITLE} mà tôi muốn bạn đặt vào trong ngữ cảnh của chương {CHAPTER_TITLE} trong {TITLE}
<chunk>
{CHUNK_CONTENT}
</chunk>
Vui lòng cung cấp một bối cảnh ngắn gọn, súc tích để đặt đoạn nội dung này trong ngữ cảnh của toàn bộ tài liệu nhằm cải thiện khả năng tìm kiếm và truy xuất đoạn nội dung. Chỉ trả lời bằng bối cảnh ngắn gọn và không thêm nội dung khác.
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

SPECIAL_CASE = ['phần', 'chương', 'mục', 'tiểu mục', 'điều']

CONTEXTUAL_CHUNK_SIZE = cfg.CONTEXTUAL_RAG.CHUNK_SIZE
CONTEXTUAL_MODEL = cfg.CONTEXTUAL_RAG.MODEL

#ORIGINAL_RAG_COLLECTION_NAME = cfg.CONTEXTUAL_RAG.ORIGIN_RAG_COLLECTION_NAME
CONTEXTUAL_RAG_COLLECTION_NAME = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME

# CONTEXTUAL_RAG_COLLECTION_NAME_1 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_1
# CONTEXTUAL_RAG_COLLECTION_NAME_2 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_2
# CONTEXTUAL_RAG_COLLECTION_NAME_3 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_3
# CONTEXTUAL_RAG_COLLECTION_NAME_4 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_4
# CONTEXTUAL_RAG_COLLECTION_NAME_5 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_5
# CONTEXTUAL_RAG_COLLECTION_NAME_6 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_6
# CONTEXTUAL_RAG_COLLECTION_NAME_7 = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME_7

QDRANT_URL = cfg.CONTEXTUAL_RAG.QDRANT_URL
ELASTIC_SEARCH_URL = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_URL
ELASTIC_SEARCH_INDEX_NAME = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME
# ELASTIC_SEARCH_INDEX_NAME_1 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_1
# ELASTIC_SEARCH_INDEX_NAME_2 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_2
# ELASTIC_SEARCH_INDEX_NAME_3 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_3
# ELASTIC_SEARCH_INDEX_NAME_4 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_4
# ELASTIC_SEARCH_INDEX_NAME_5 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_5
# ELASTIC_SEARCH_INDEX_NAME_6 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_6
# ELASTIC_SEARCH_INDEX_NAME_7 = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME_7



NUM_CHUNKS_TO_RECALL = cfg.CONTEXTUAL_RAG.NUM_CHUNKS_TO_RECALL
SEMANTIC_WEIGHT = cfg.CONTEXTUAL_RAG.SEMANTIC_WEIGHT
BM25_WEIGHT = cfg.CONTEXTUAL_RAG.BM25_WEIGHT
TOP_N = cfg.CONTEXTUAL_RAG.TOP_N

SUPPORTED_FILE_EXTENSIONS = [".pdf"]

# Agent configuration
AGENT_TYPE = cfg.AGENT.TYPE