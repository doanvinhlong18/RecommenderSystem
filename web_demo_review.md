# Web Demo Review (Backend + Frontend)

## Mục tiêu và phạm vi
Tài liệu này review chi tiết phần web demo của dự án, gồm:
- Backend API demo: `web/backend/api_server.py`
- Frontend UI: `web/frontend/index.html`, `web/frontend/app.js`, `web/frontend/styles.css`

Review theo mindset code review: ưu tiên bug/rủi ro/regression/testing gaps trước, sau đó mới tóm tắt kiến trúc và đề xuất triển khai.

---

## Findings (ưu tiên theo mức độ nghiêm trọng)

### 1) [High] Sai lệch đường dẫn model giữa train và web backend
- **Bằng chứng**:
  - Backend demo load model từ `MODELS_DIR / "hybrid"` tại `web/backend/api_server.py:146`.
  - Pipeline train hiện tại lưu engine vào `MODELS_DIR / "learned_hybrid"` tại `train.py:558`.
- **Tác động**:
  - Web có thể khởi động "thành công" nhưng load sai artifact, hoặc không load được model mới nhất.
  - Dẫn đến kết quả recommend không đồng bộ với pipeline train/evaluation.
- **Khuyến nghị**:
  - Chuẩn hóa 1 nguồn duy nhất cho model dir (ưu tiên `learned_hybrid`).
  - Hỗ trợ fallback có thứ tự rõ ràng (`learned_hybrid` -> `hybrid`) và log cảnh báo rõ.

### 2) [High] Blocking I/O trong async endpoint lịch sử user
- **Bằng chứng**:
  - Endpoint async `get_user_history` tại `web/backend/api_server.py:539`.
  - Gọi trực tiếp hàm đọc CSV đồng bộ `_read_animelist_history_csv(...)` tại `web/backend/api_server.py:581`.
  - Hàm đọc CSV có vòng lặp scan file và parse đồng bộ tại cuối file.
- **Tác động**:
  - Có khả năng block event loop khi request đồng thời (nhất là lúc nhiều người bấm For You).
  - Tiếp diện có thể thấy "trễ"/timeout cảm nhận dù backend vẫn sống.
- **Khuyến nghị**:
  - Chuyển đọc CSV sang thread pool (`run_in_executor`) hoặc tiền xử lý user-history index.
  - Bổ sung cache theo user với TTL + giới hạn kích thước.

### 3) [Medium] Cơ chế giới hạn scan lịch sử có thể trả false negative
- **Bằng chứng**:
  - Giới hạn scan: `max_rows = 300_000` và `time_budget_s = 1.0` tại `web/backend/api_server.py:1084-1085`.
- **Tác động**:
  - User hợp lệ nhưng nằm sau file có thể bị trả "không có history".
  - Kết quả không ổn định theo thứ tự dòng trong CSV, ảnh hưởng độ tin cậy demo.
- **Khuyến nghị**:
  - Dùng secondary index theo `user_id` (build một lần).
  - Nếu cần timeout cứng, trả về có `partial=true` để frontend biết dữ liệu chưa đầy đủ.

### 4) [Medium] Cấu hình CORS mở quá rộng và dễ gây nhầm hành vi credentials
- **Bằng chứng**:
  - `allow_origins=["*"]` tại `web/backend/api_server.py:191`.
  - `allow_credentials=True` tại `web/backend/api_server.py:192`.
- **Tác động**:
  - Rủi ro bảo mật khi deploy public.
  - Browser có thể hành xử bất ngờ với credentials + wildcard origin.
- **Khuyến nghị**:
  - Khóa origin theo môi trường (`localhost`, domain frontend thật sự).
  - Tắt credentials nếu không dùng cookie/session cross-site.

### 5) [Medium] Độ lệch hợp đồng DOM: JS render history/chart nhưng HTML không còn node
- **Bằng chứng**:
  - JS tìm `#userHistorySection` tại `web/frontend/app.js:783,1091`.
  - JS tìm `#historyChartSection` tại `web/frontend/app.js:870,1079`.
  - JS tìm `#recsChartSection` tại `web/frontend/app.js:965,1081`.
  - HTML hiện tại không có các id trên (`web/frontend/index.html`, không tìm thấy).
- **Tác động**:
  - Tính năng history/chart "có code" nhưng không hiển thị thật sự.
  - Team dễ nhầm rằng đã support UX này trong khi UI đang tắt logic bằng cách loại node.
- **Khuyến nghị**:
  - Chốt 1 trong 2 hướng:
    1. Xóa dead-code phần history/chart trong JS/CSS.
    2. Đưa lại section HTML tương ứng và test E2E.

### 6) [Medium] Không có test tự động cho web demo
- **Bằng chứng**:
  - Không tìm thấy file test trong `web/` (không có `web/**/*test*`).
- **Tác động**:
  - Dễ bị vỡ silently khi đổi endpoint contract, id DOM, hoặc shape response.
- **Khuyến nghị**:
  - Thêm test contract API (FastAPI TestClient).
  - Thêm smoke test frontend (Playwright/Cypress) cho 4 luồng: Search, Popular, For You, Compare.

### 7) [Low] Cache frontend/backend chưa có cơ chế giới hạn kích thước
- **Bằng chứng**:
  - Frontend cache (`state.cache.search/recommendations/popular/images`) ở `web/frontend/app.js:21-31`.
  - Backend cache `_user_history_cache` khai báo global ở `web/backend/api_server.py` (phần đầu file).
- **Tác động**:
  - Chạy lâu có thể tăng RAM theo thời gian.
- **Khuyến nghị**:
  - Dùng LRU/TTL cache (giới hạn entries, expire theo thời gian).

### 8) [Low] Frontend phụ thuộc API ngoài Jikan cho ảnh
- **Bằng chứng**:
  - Gọi `https://api.jikan.moe/v4` tại `web/frontend/app.js:11,289-311`.
- **Tác động**:
  - Ảnh hưởng tốc độ render và tính ổn định nếu Jikan rate limit/giật.
- **Khuyến nghị**:
  - Ưu tiên image URL từ backend trước, Jikan chỉ fallback.
  - Có retry/backoff nhẹ + placeholder nhất quán.

---

## Open questions / assumptions
- Web demo hiện tại là bản "showcase" hay đã hướng tới production-lite?
- Có chủ trương giữ `HybridEngine` cũ trong demo hay đồng bộ sang `LearnedHybridEngine`?
- Feature history/chart còn trong scope không, hay đã quyết định ẩn hoàn toàn?

---

## Tổng quan kiến trúc web demo

### 1) Backend (`web/backend/api_server.py`)
- **Lifespan startup**: load model 1 lần khi app khởi động.
- **REST groups**:
  - Status/health: `/health`, `/api/status`
  - Search: `/api/search`, `/api/autocomplete`
  - Anime detail: `/api/anime/{id}`, `/api/anime/name/{name}`
  - Recommendation: `/api/recommend/anime/*`, `/api/recommend/user/{user_id}`
  - User history: `/api/user/{user_id}/history`
  - Demo UX: `/api/demo/users`, `/api/popular`, `/api/compare`, `/api/weights`, `/api/genres`
- **Điểm mạnh**:
  - API bao phủ toàn bộ luồng demo.
  - Có fallback cho demo user/history để UI không chết cứng.

### 2) Frontend (`web/frontend/index.html`, `web/frontend/app.js`, `web/frontend/styles.css`)
- **SPA nhẹ theo page sections**: Search, Popular, For You, Compare.
- **State management đơn giản**: object `state` + cache client-side.
- **UX logic**:
  - Overlay loading + toast notifications.
  - Autocomplete + quick tags.
  - Compare nhiều phương pháp cùng lúc.
- **Điểm mạnh**:
  - UI đẹp, rõ ràng, có nhiều luồng trải nghiệm.
  - Có phòng thủ timeout/phòng thủ null DOM ở nhiều chỗ.

### 3) Luồng dữ liệu chính (For You)
1. User chọn/bấm demo user -> `handleUserRecommendations`.
2. Frontend gọi history (`/api/user/{id}/history`) trước.
3. Frontend gọi recommend user (`/api/recommend/user/{id}`) sau.
4. Render strategy + grid recommendations + (nếu có node) charts/history.

---

## Đề xuất ưu tiên implement (roadmap ngắn)
1. **P0**: Đồng bộ model dir load/save (`learned_hybrid`) + thêm logging rõ artifact đang sử dụng.
2. **P0**: Tách đọc history CSV khỏi event loop (executor) + trả response có `partial` nếu timeout.
3. **P1**: Quyết định rõ feature history/chart (giữ và đưa lại HTML, hoặc xóa dead-code).
4. **P1**: Thêm test contract cho endpoint quan trọng (`/api/status`, `/api/recommend/user/*`, `/api/user/*/history`).
5. **P2**: Bổ sung LRU/TTL cho cache frontend/backend.
6. **P2**: Hardening CORS theo env deploy.

---

## Phần trình bày (slide-ready)

### A. Mục tiêu bài thuyết trình (1 slide)
- Giới thiệu web demo recommendation anime đã tích hợp 4 nhóm model.
- Trình bày kiến trúc FE/BE, luồng dữ liệu, và giá trị cho người dùng.
- Nêu rõ các rủi ro kỹ thuật hiện tại và kế hoạch nâng cấp.

### B. Nội dung đề xuất 8 slide
1. **Problem & Goal**
   - Người dùng khó tìm anime phù hợp trong kho nội dung lớn.
   - Demo giải bài toán tìm anime tương đồng + gợi ý cá nhân hóa.
2. **System Architecture**
   - Frontend SPA (Search/Popular/For You/Compare).
   - Backend FastAPI + hybrid recommendation engine.
3. **Core Features**
   - Search + autocomplete.
   - Similar anime theo nhiều method.
   - For You recommendations + demo users.
   - Compare methods + weight tuning.
4. **Data Flow (For You)**
   - User input -> history -> recommendation -> render cards/charts.
5. **Strengths**
   - UX rõ ràng, tốc độ cảm nhận tốt, fallback cho demo.
   - Có endpoint phong phú để demo cho business/stakeholder.
6. **Technical Risks (thang điểm severity)**
   - Model path mismatch.
   - Blocking I/O trong async endpoint.
   - DOM contract drift, thiếu test web.
7. **Improvement Plan (30-60-90 days)**
   - 30 ngày: fix P0.
   - 60 ngày: test automation + history architecture.
   - 90 ngày: hardening và observability.
8. **Expected Impact**
   - Giảm timeout/trễ UI.
   - Tăng tính ổn định demo.
   - Tăng độ tin cậy kết quả recommend khi trình bày cho stakeholder.

### C. Script thuyết trình ngắn (5-7 phút)
- "Hệ thống web demo này mô phỏng đầy đủ hành trình khám phá anime: tìm kiếm, xem xu hướng, nhận gợi ý cá nhân hóa, và so sánh các thuật toán recommendation."
- "Về kiến trúc, frontend SPA giao tiếp FastAPI backend qua các endpoint chức năng. Backend kết nối hybrid engine gồm content, collaborative, implicit và popularity."
- "Điểm mạnh là trải nghiệm người dùng rất trực quan, tính năng compare và weight tuning giúp giải thích kết quả recommendation dễ dàng."
- "Qua review kỹ thuật, có 3 rủi ro ưu tiên cao: sai lệch đường dẫn model giữa train và serve, doc CSV đồng bộ trong async endpoint history, và do lệch giữa JS và HTML ở phần history/chart."
- "Roadmap đề xuất là fix P0 ngay để ổn định demo, tiếp theo bổ sung test contract/API + E2E, sau đó hardening cache/CORS và observability."
- "Kết quả kỳ vọng là UI ổn định hơn, giảm timeout, và demo tin cậy hơn khi trình bày cho stakeholder hoặc chuyển sang production-lite."

### D. Demo checklist khi trình bày
- Kiểm tra `/api/status` = ready.
- Demo Search -> chọn 1 anime -> Get Recommendations (hybrid).
- Chuyển tab Popular -> đổi tab top_rated/trending.
- Tab For You -> bấm 1 demo user -> show recommendation.
- Tab Compare -> nhập anime -> compare content/collaborative/hybrid.
- Nếu cần, demo update weights và kết quả thay đổi.

---

## Kết luận
Web demo hiện tại đã đạt mức "demo được, dễ hiểu, dễ trình bày". Tuy nhiên để đạt mức "ổn định cho trình diễn thường xuyên", cần ưu tiên xử lý 3 điểm P0/P1 nếu trên trước khi mở rộng thêm tính năng.
