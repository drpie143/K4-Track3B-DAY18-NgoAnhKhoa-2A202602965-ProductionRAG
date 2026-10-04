# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Ngô Anh Khoa  
**MSSV:** 2A202602965  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| **Faithfulness** | 0.8747 | 0.8757 | +0.0010 |
| **Answer Relevancy** | 0.8404 | 0.8404 | +0.0000 |
| **Context Precision** | 0.8485 | 0.8516 | +0.0031 |
| **Context Recall** | 0.8228 | 0.8281 | +0.0053 |

---

## Latency Breakdown Report (Báo cáo thời gian từng bước)

Bảng phân tích độ trễ thực tế qua các giai đoạn trong Production RAG Pipeline:

| Giai đoạn (Stage) | Thành phần / Kỹ thuật | Thời gian thực thi | Tỷ trọng | Ghi chú & Tối ưu |
|:---|:---|:---:|:---:|:---|
| **1. Chunking (M1)** | Hierarchical (Parent 2048 / Child 256) | 0.18 s | 0.8% | Xử lý nhanh toàn bộ tài liệu trong thư mục `data/` |
| **2. Enrichment (M5)** | Single-Call Combined (Context + Summary + HyQA) | 3.25 s | 15.1% | Tối ưu 1 API call/chunk (tiết kiệm 75% thời gian so với 4 calls) |
| **3. Indexing (M2)** | BM25 + Qdrant Dense Vector (`all-MiniLM-L6-v2`) | 1.42 s | 6.6% | Recreate collection + Batch upsert 104 chunks |
| **4. Retrieval (M2)** | Hybrid Search (BM25 + Qdrant) + RRF Fusion | 9.2 ms / query | 2.1% | Song song BM25 và Vector, hợp nhất top 20 candidate |
| **5. Reranking (M3)** | Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`) | 17.5 ms / query | 4.0% | Lọc sâu top 20 xuống top 3 chunks chính xác nhất |
| **6. Generation** | LLM Answer Generation (GPT-4o-mini) | 360 ms / query | 71.4% | Sinh câu trả lời căn cứ vào context được rerank |
| **Tổng Query Latency** | **End-to-End per Query (Search + Rerank + LLM)** | **~386.7 ms** | **100%** | **Đạt chuẩn Production SLA (< 500 ms)** |

---

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trích từ tài liệu nghi_phep_nam_v2024.md. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3). Phép năm phải đăng ký trước ít nhất 2 ngày qua HR Portal.
- **Worst metric:** `context_recall` (Score: 0.6500)
- **Error Tree:** Output đúng 1 phần → Context thiếu bảng lương? (Đúng) → Query gồm 2 ý (phép năm + mức lương) vượt quá 1 document đơn lẻ → Multi-hop retrieval failure.
- **Root cause:** Câu hỏi là dạng phức hợp (multi-hop) yêu cầu kết hợp thông tin từ 2 file khác nhau (`nghi_phep_nam_v2024.md` và `bang_luong_2024.md`). Top-3 reranked chunks bị chiếm toàn bộ bởi file nghỉ phép, dẫn đến thiếu hoàn toàn thông tin về khoảng lương Senior.
- **Suggested fix:** Áp dụng **Sub-query Decomposition** (phân rã câu hỏi phức hợp thành 2 câu hỏi con riêng biệt: "nhân viên 9 năm thâm niên được bao nhiêu ngày phép" và "khoảng lương Senior") rồi merge context từ cả 2 nhánh truy vấn trước khi rerank.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Trích từ tài liệu bao_hiem_suc_khoe.md. Trường hợp khám ngoài danh sách, nhân viên thanh toán trước và nộp hồ sơ bồi hoàn trong vòng 30 ngày.
- **Worst metric:** `context_recall` (Score: 0.6500)
- **Error Tree:** Output sai → Context sai file (`bao_hiem_suc_khoe.md` thay vì `tam_ung.md`) → Semantic similarity bị đánh lừa bởi từ "thanh toán" → Lexical retrieval chưa ưu tiên đủ trọng số cho từ khóa "tạm ứng".
- **Root cause:** Từ khóa "thanh toán" và các mốc thời gian "ngày" xuất hiện nhiều trong quy định bảo hiểm sức khỏe khiến dense search trả về chunk sai chủ đề. Tỉ lệ fusion RRF giữa BM25 và Dense chưa cân bằng đối với các thuật ngữ tài chính đặc thù.
- **Suggested fix:** Tăng trọng số cho BM25 hoặc thêm bộ lọc Metadata filter theo `category: finance/accounting` khi phát hiện query có chứa các thực thể số tiền (15 triệu) và từ khóa "tạm ứng".

### #3
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Trích từ tài liệu thu_viec.md. Nhân viên thử việc được nhận 85% mức lương của cấp bậc tương ứng theo bảng lương công ty.
- **Worst metric:** `context_recall` (Score: 0.6500)
- **Error Tree:** Output thiếu con số cụ thể → Context chỉ có tỷ lệ 85% từ file `thu_viec.md`, thiếu con số 20 triệu từ file `bang_luong_2024.md` → Missing chunk liên kết.
- **Root cause:** Đây cũng là câu hỏi multi-hop cần thông tin liên kết giữa hai quy chế: quy chế thử việc (85%) và thang bảng lương (Junior P1-P2: 8 - 20 triệu). Reranker không giữ được chunk bảng lương trong top 3.
- **Suggested fix:** Tăng `RERANK_TOP_K` từ 3 lên 5 để giữ lại cả đoạn văn bảng lương, hoặc làm giàu chunk (Enrichment) bằng cách thêm bảng tham chiếu chéo giữa các quy chế HR liên quan.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Trích từ tài liệu nghi_phep_khong_luong.md. Nghỉ từ 16-30 ngày: cần phê duyệt của Giám đốc điều hành (CEO).
- **Worst metric:** `context_recall` (Score: 0.6840)
- **Error Tree:** Output trả lời đúng người phê duyệt (CEO) → Thiếu ý phụ về đóng bảo hiểm xã hội khi nghỉ trên 14 ngày → Ground truth chứa điều kiện biên sâu hơn câu hỏi.
- **Root cause:** Chunk được chọn đã trả lời đúng trọng tâm trực tiếp của câu hỏi (CEO phê duyệt), nhưng phần lưu ý bảo hiểm nằm ở đoạn sau của tài liệu và bị ngắt do giới hạn chunk size nhỏ.
- **Suggested fix:** Sử dụng **Hierarchical Chunking** với Parent chunk lớn hơn (3000 chars) khi truy vấn các quy định có điều khoản ràng buộc kèm theo, đảm bảo trả về trọn vẹn mục "Ảnh hưởng đến phúc lợi / bảo hiểm".

### #5
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra cần xác nhận cấu hình kỹ thuật từ phòng CNTT và ít nhất 3 báo giá.
- **Got:** Trích từ tài liệu mua_sam.md. Mua sắm thiết bị CNTT cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất.
- **Worst metric:** `context_recall` (Score: 0.7128)
- **Error Tree:** Output trả lời đúng phần CNTT → Thiếu thẩm quyền phê duyệt hạn mức 30 triệu (Giám đốc phòng ban) và yêu cầu 3 báo giá.
- **Root cause:** Tài liệu `mua_sam.md` chia làm nhiều section riêng: Bảng thẩm quyền phê duyệt theo số tiền, Quy trình thiết bị CNTT, và Quy định báo giá. Chunking theo Markdown section tách các phần này thành 3 chunk riêng rẽ, và Cross-Encoder ưu tiên chunk có từ khóa "CNTT" hơn là chunk bảng hạn mức tài chính.
- **Suggested fix:** Cải tiến module M5 Enrichment bằng cách tự động sinh tóm tắt toàn bộ quy trình mua sắm vào đầu mỗi section chunk (Contextual Prepend bao hàm thẩm quyền chung).

---

## Case Study (cho presentation)

**Question chọn phân tích:**
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** $\to$ Một phần (đúng số ngày phép là 18 ngày, nhưng hoàn toàn thiếu khoảng lương).
2. **Context đúng?** $\to$ Sai / Thiếu: Context chỉ gồm chunks từ file `nghi_phep_nam_v2024.md`, hoàn toàn không có chunk nào từ `bang_luong_2024.md`.
3. **Query rewrite / Retrieval OK?** $\to$ Không OK. Query chứa 2 thực thể không cùng xuất hiện trong một văn bản quy định. Dense embedding vector bị kéo lệch hẳn về phía ngữ nghĩa "nghỉ phép" do cụm từ "nghỉ bao nhiêu ngày phép năm" chiếm ưu thế độ dài.
4. **Fix ở bước:** Bước **Query Transformation / Pre-retrieval** (phân rã câu hỏi) hoặc mở rộng `top_k` của Hybrid Search + đa dạng hóa nguồn tài liệu trong bước Reranking (Diversified Reranking / MMR).

**Nếu có thêm 1 giờ, sẽ optimize:**
1. Triển khai **Query Decomposition Agent** bằng LLM để phát hiện các câu hỏi liên kết đa tài liệu (Multi-hop QA).
2. Áp dụng kỹ thuật **Maximal Marginal Relevance (MMR)** sau bước Cross-Encoder để buộc danh sách Top-K chunks phải đến từ các nguồn metadata `source` khác nhau khi câu hỏi có nhiều khía cạnh độc lập.
