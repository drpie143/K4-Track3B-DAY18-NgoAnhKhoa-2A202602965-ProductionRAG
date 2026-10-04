# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Ngô Anh Khoa  
**MSSV:** 2A202602965  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng ánh xạ các khái niệm lý thuyết từ bài giảng vào mã nguồn thực tế đã triển khai trong Lab 18:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Advanced Chunking** (Semantic, Hierarchical, Structure-aware) | M1 | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()` | `chunk_hierarchical` là lựa chọn tối ưu cho production RAG: tách parent chunks (2048 chars) và child chunks (256 chars) có liên kết `parent_id`. Khi search dùng child để đạt precision cao, khi sinh câu trả lời trả về parent để LLM có đầy đủ ngữ cảnh bao quát, không bị ngắt giữa ý. `chunk_semantic` dùng cosine similarity giữa các câu liên tiếp tránh cắt ngang luận điểm. |
| **Hybrid Search & Fusion** (BM25 + Dense + RRF) | M2 | `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | BM25 giải quyết điểm yếu của Dense Search khi truy vấn các từ khóa định danh, số liệu cụ thể (VD: "12 ngày", "90 ngày"). RRF ($k=60$) kết hợp thứ hạng của 2 phương pháp mà không cần chuẩn hóa scale điểm, cho kết quả cân bằng giữa semantic và lexical match. Tiếng Việt cần qua `underthesea.word_tokenize` và thay `_` bằng khoảng trắng để BM25 bắt đúng token. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Bi-encoder nhanh trong việc lọc top-20 ứng viên từ hàng nghìn documents, nhưng Cross-Encoder (`BAAI/bge-reranker-v2-m3`) thực hiện full cross-attention giữa `(query, document)` giúp đẩy các đoạn chứa trực tiếp câu trả lời lên top 3, loại bỏ hoàn toàn các đoạn gây nhiễu trước khi nạp vào LLM prompt. |
| **RAGAS Evaluation & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Đánh giá toàn diện 4 khía cạnh: Faithfulness (LLM hallucination), Answer Relevancy (đúng trọng tâm), Context Precision (tỉ lệ chunk liên quan ở top đầu), Context Recall (độ đầy đủ của context so với ground truth). Cây chẩn đoán Diagnostic Tree giúp xác định lỗi thuộc về khâu Retrieval hay Generation. |
| **Contextual Enrichment** | M5 | `contextual_prepend()`, `_enrich_single_call()` | Kỹ thuật Contextual Prepend bổ sung nguồn tài liệu và tóm lược vị trí của chunk giúp vector embedding không bị mất bối cảnh (giảm tới 49% retrieval failure theo Anthropic). Tối ưu thành 1 single LLM call gộp (Summary + HyQA + Prepend + Metadata) vừa giảm 75% chi phí API vừa tăng tốc độ indexing. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

1. **Vấn đề Tokenization tiếng Việt với BM25:**
   - **Lỗi gặp phải:** Truy vấn BM25 trả về score = 0 cho các câu có từ khóa chính xác như "nghỉ phép", dù tài liệu có chứa từ này.
   - **Nguyên nhân gốc rễ:** `underthesea.word_tokenize(format="text")` tự động nối từ ghép bằng dấu gạch dưới (ví dụ: `nghỉ_phép`). Nhưng hàm tách từ của BM25 cắt theo dấu cách (`split(" ")`), trong khi query người dùng nhập là `nghỉ phép` (2 token riêng rẽ), dẫn đến không khớp với token `nghỉ_phép`.
   - **Cách debug & xử lý:** Chuẩn hóa đầu ra của `word_tokenize` bằng cách gọi `.replace("_", " ")`. Nhờ đó từ ghép tiếng Việt vừa được phân đoạn ngữ pháp vừa đảm bảo tính tương thích với BM25Okapi.

2. **Xung đột phiên bản Qdrant-Client API:**
   - **Lỗi gặp phải:** Gọi `client.search()` bị cảnh báo deprecated hoặc lỗi AttributeError trên `qdrant-client >= 2.0`.
   - **Cách debug & xử lý:** Đọc tài liệu mới nhất của Qdrant Python SDK, chuyển đổi sang dùng method `self.client.query_points(collection_name=..., query=query_vector, limit=top_k)` và truy xuất kết quả qua `response.points`. Thêm khối `try/except` fallback để tương thích ngược.

3. **Tương thích Model Cross-Encoder trên Transformers:**
   - **Lỗi gặp phải:** Dùng thư viện `FlagEmbedding.FlagReranker` bị lỗi với phiên bản `transformers >= 5.0` (lỗi XLMRobertaTokenizer).
   - **Cách xử lý:** Thay thế trực tiếp bằng `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")`, vừa nhẹ, vừa tương thích hoàn toàn với PyTorch và Hugging Face Hub hiện đại.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy chế Doanh nghiệp (Enterprise Policy AI Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=100) và vector database dùng embedding OpenAI text-embedding-3-small.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Context Precision thấp:* Khi nhân viên hỏi về các chính sách có sự thay đổi qua các năm (VD: Quy chế nghỉ phép 2023 vs 2024), hệ thống thường truy xuất lẫn lộn các điều khoản cũ.
  - *Từ khóa chính xác bị trượt:* Khi tìm kiếm theo số quyết định, số thông tư, tên quy trình nội bộ thì dense embedding trả về kết quả mờ nhạt.
  - *Hallucination:* LLM tự suy diễn các mốc thời hạn nếu context bị cắt ngang giữa các điều luật.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Chuyển sang **Hierarchical Chunking** kết hợp **Structure-aware**:
   - Tách theo cấu trúc Chương / Điều / Khoản (dựa trên tiêu đề Markdown hoặc regex văn bản pháp quy).
   - Thiết lập Parent chunk (toàn bộ Điều khoản - ~1500 chars) và Child chunk (từng Khoản/Điểm - ~250 chars) có liên kết ID để khi tìm kiếm đạt độ chính xác đến từng dòng nhưng khi tổng hợp câu trả lời vẫn nắm trọn bối cảnh pháp lý của Điều luật.
2. **Search retrieval:** Triển khai **Hybrid Search (BM25 + Dense BAAI/bge-m3 + RRF)**:
   - BM25 với bộ tách từ tiếng Việt đảm bảo bắt chính xác 100% số điều, số hiệu văn bản, định mức tiền thưởng.
   - Dense bge-m3 hỗ trợ đa ngôn ngữ và ngữ nghĩa sâu sắc cho câu hỏi tự nhiên.
   - Hợp nhất bằng RRF ($k=60$) lấy top-20 ứng viên.
3. **Reranking:** Tích hợp **Cross-Encoder (`BAAI/bge-reranker-v2-m3`)**:
   - Lọc từ top-20 xuống top-3 trước khi nạp cho LLM. Đảm bảo triệt tiêu các điều khoản đã hết hiệu lực hoặc không liên quan trực tiếp.
4. **Enrichment:** Áp dụng **Contextual Prepend & Auto Metadata**:
   - Prepend tên văn bản, năm ban hành, phòng ban phụ trách vào đầu mỗi chunk.
   - Thêm metadata filter theo trạng thái `status: active` và `year: 2024` để loại trừ tài liệu cũ khi người dùng không hỏi lịch sử.
5. **Evaluation:** Thiết lập hệ thống benchmark tự động định kỳ bằng **RAGAS (4 metrics)** trên tập Golden Test Set 100 câu hỏi nghiệp vụ, giám sát regression qua CI/CD.

#### 3. Timeline triển khai
- **Tuần 1:** Refactor pipeline chunking (Hierarchical + Structure) và re-index dữ liệu với Hybrid Search (BM25 + Qdrant Dense).
- **Tuần 2:** Tích hợp Cross-Encoder Reranker, kiểm thử độ trễ latency trên GPU/CPU và tối ưu batch size.
- **Tuần 3:** Tự động hóa Enrichment quy chế bằng GPT-4o-mini (single call) và chạy RAGAS evaluation đối sánh, tối ưu prompt template cho generation.
