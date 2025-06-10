from config.config import get_config

cfg = get_config("/workspace/competitions/Sly/Duy_NCKH_2025_dev/config/config.yaml")


CUSTOM_REFINE_PROMPT =  (
    "Câu hỏi ban đầu như sau: {query_str}\n"
    "Chúng ta hiện đang có câu trả lời tạm thời như sau:\n"
    "-------------------\n"
    "{existing_answer}\n"
    "-------------------\n"
    "Dưới đây là ngữ cảnh pháp lý mới được bổ sung:\n"
    "-------------------\n"
    "{context_msg}\n"
    "-------------------\n"
    "Dựa vào thông tin mới, bạn hãy điều chỉnh lại câu trả lời trên **nếu cần thiết**, và đảm bảo tuân thủ cấu trúc trả lời như sau:\n"
    "- Nhắc lại câu hỏi.\n"
    "- Trả lời thẳng vào câu hỏi.\n"
    "- Ghi rõ tên điều/khoản/văn bản pháp luật trong nội dung tham chiếu.\n"
    "- Trích dẫn nội dung đầy đủ của đoạn luật tương ứng.\n"
    "Nếu thông tin mới không hữu ích, hãy giữ nguyên câu trả lời cũ.\n"
    "Câu trả lời đã chỉnh sửa (nếu có):"
)

CUSTOM_OPENAI_SUB_QUESTION_PROMPT = """\
You are a world class state of the art agent.

You have access to multiple tools, each representing a different data source or API.
Each of the tools has a name and a description, formatted as a JSON dictionary.
The keys of the dictionary are the names of the tools and the values are the \
descriptions.
Your purpose is to help answer a complex user question by generating a list of sub \
questions that can be answered by the tools.

These are the guidelines you consider when completing your task:
* Be as specific as possible
* The sub questions should be relevant to the user question
* The sub questions should be answerable by the tools provided
* You can generate multiple sub questions for each tool
* Tools must be specified by their name, not their description
* You don't need to use a tool if you don't think it's relevant

## Example
**Query:** Tôi không đội nón bảo hiểm khi tham gia giao thông và gây tai nạn thì bị phạt thế nào?
- **Sub-question 1:** Hành vi không đội nón bảo hiểm khi tham gia giao thông bị phạt như thế nào?
- **Sub-question 2:** Nếu không đội nón bảo hiểm và gây tai nạn thì sẽ bị phạt ra sao?


Output the list of sub questions by calling the SubQuestionList function.

## Tools
```json
{tools_str}
```

## User Question
{query_str}
"""

QA_PROMPT = (
    "Dưới đây là một nội dung của một Điều/hoặc những Khoản nhỏ của một điều trích từ một thông tư/nghị định/luật: \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Dựa vào thông tin trên, hãy trả lời câu hỏi sau: {query_str}\n\
    Lưu ý luôn luôn suy nghĩ kĩ trước khi trả lời. Để trả lời câu hỏi vế đầu tiên bạn phải trích dẫn lại ý chính của câu hỏi và trả lời thẳng vào câu hỏi kèm theo tên điều, khoản.\
    Sau đó trích dẫn lại nội dung bạn đã tham chiếu (tên điều, khoản). Nếu không biết hãy trả lời thành thật và không đưa ra thông tin sai lệch"
)

# Contextual RAG configuration
EMBEDDING_MODEL = cfg.CONTEXTUAL_RAG.EMBEDDING_MODEL
CONTEXTUAL_MODEL = cfg.CONTEXTUAL_RAG.MODEL

CONTEXTUAL_RAG_COLLECTION_NAME = cfg.CONTEXTUAL_RAG.CONTEXTUAL_RAG_COLLECTION_NAME

QDRANT_URL = cfg.CONTEXTUAL_RAG.QDRANT_URL
ELASTIC_SEARCH_URL = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_URL
ELASTIC_SEARCH_INDEX_NAME = cfg.CONTEXTUAL_RAG.ELASTIC_SEARCH_INDEX_NAME

SEMANTIC_WEIGHT = cfg.CONTEXTUAL_RAG.SEMANTIC_WEIGHT
BM25_WEIGHT = cfg.CONTEXTUAL_RAG.BM25_WEIGHT

SUPPORTED_FILE_EXTENSIONS = [".pdf"]

