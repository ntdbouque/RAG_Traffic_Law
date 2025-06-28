# Traffic Law RAG on decree 168/2024 System

---

## Pipeline Overview:
![Pipeline](/asset/DACS2/DACS2-overview-pipeline.drawio.png)

---

## 📁 Constructed Data:

- `sample/output_with_full_article_content.csv`: 

---

## Setup API key: 
Please create `.env` file and provide these API keys:

|         NAME          |                     Where to get ?                      |
| :-------------------: | :-----------------------------------------------------: |
|   `OPENAI_API_KEY`    | [OpenAI Platform](https://platform.openai.com/api-keys) |

## Setup Elasticsearch and Qdrant Client
```bash
docker compose up -d
```

## Requirement:
```bash
pip install -r requirements.txt
```

## Usage: 

### 1. `run/run_ingest_from_csv.py` – Thêm ngữ cảnh từ tệp CSV đã được thêm ngữ cảnh

Script này dùng để ingest data đã được xử lí vào file csv

```bash
python run/run_ingest_from_csv.py \
  --csv_folder your_folder_contain_csv_file \
```
**Parameter:**
`--csv_folder`: đường dẫn đến folder chứa các file csv cần ingest

### 2. `run_generating_qa.py` 
Script này dùng để tạo ra bộ dataset qa để đánh giá khả năng truy vấn

```bash
python run/run_generating_qa.py \
 --output_path path_to_save_your_qa.json\
 --num_questions 2
```
**Parameters:**
- `--output_path`: đường dẫn để lưu bộ question-answering dataset 
- `--num_questions`: số lượng câu hỏi được tạo ra từ mỗi chunk

### 3. `evaluator/run_retrieval_evaluation.ipynb`
Tham khảo notebook trên để đánh giá `retriever` trên một sample và toàn bộ dataset

### 4. **Run `app.py`:**
```bash
uvicorn app:app --host 0.0.0.0 --port your_port_here --loop asyncio
```

### 5. Example Usage:

- **`test_retrieval.py`:** 
```python
from source.rag.retrieval import RetrievalPipeline
from source.settings import Settings

query = 'người được chở trên xe máy mà sử dụng ô dù thì bị phạt thế nào?'
retriever = RetrievalPipeline()
response = retriever.retrieve(query)
```

- **`test_query_engine.py`:**
```python 
retriever = RetrievalPipeline()
llm = OpenAI(
                model=setting.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                logprobs=None,
                default_headers={},
            )
synthesizer = get_response_synthesizer(response_mode="compact")


query = 'Tôi lái xe hơi mà trong hơi thở có nồng độ cồn thì sao?'
my_query_engine = MyQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=PromptTemplate(QA_PROMPT),
)
response = my_query_engine.query(query)
```
- **`ingest.py:`**
```bash
cd source/rag
python ingest.py
```


