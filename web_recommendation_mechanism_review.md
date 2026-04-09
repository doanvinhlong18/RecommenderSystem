# Phân Tích Chi Tiết Cơ Chế Recommend Trong Web Demo

## 1) Mục tiêu tài liệu
Tài liệu này phân tích chi tiết:
- Cơ chế recommendation của từng phương thức đang dùng trong web demo.
- Lý do vì sao mỗi màn hình chọn (hoặc ưu tiên) phương thức đó.
- Luồng dữ liệu từ Frontend -> API -> Hybrid Engine -> Sub-model.

Phạm vi code đối chiếu:
- `web/frontend/index.html`
- `web/frontend/app.js`
- `web/backend/api_server.py`
- `models/hybrid/__init__.py`
- `models/hybrid/learned_hybrid.py`
- `models/content/content_based.py`
- `models/collaborative/matrix_factorization.py`
- `models/implicit/als_implicit.py`
- `models/popularity/popularity_model.py`

---

## 2) Kiến trúc recommendation thực tế trong web

### 2.1 Điểm quan trọng: `HybridEngine` trong web thực chất là `LearnedHybridEngine`
Trong `models/hybrid/__init__.py`, `HybridEngine` được alias sang `LearnedHybridEngine`.
Điều này có nghĩa:
- Backend gọi `from models.hybrid import HybridEngine` nhưng thực thi logic của `LearnedHybridEngine`.
- Mọi endpoint recommendation trong web đang chạy theo cơ chế cascade + meta-model (khi có model đã train).

### 2.2 Luồng tổng quát
1. Frontend gọi endpoint theo màn hình.
2. Backend nhận request tại `web/backend/api_server.py`.
3. Backend gọi `hybrid_engine.recommend_similar_anime(...)` hoặc `hybrid_engine.recommend_for_user(...)`.
4. `LearnedHybridEngine` phối hợp các sub-model:
   - Content-based
   - Collaborative (BPR/SVD wrapper)
   - Implicit ALS
   - Popularity
5. Kết quả được enrich thêm metadata + image URL rồi trả về UI.

---

## 3) Cơ chế của từng phương thức

## 3.1 Content-Based

### Cách hoạt động
- Nguồn chính: embedding nội dung anime (đã chuẩn hóa L2) trong `ContentBasedRecommender`.
- `get_similar_anime(...)`:
  - Dùng FAISS (nếu có) để tìm hàng xóm gần nhất theo cosine/IP.
  - Fallback sang nhân ma trận `numpy @` nếu không có FAISS.
- `recommend_for_user(...)`:
  - Dựng user vector từ các anime user đã chấm (`build_user_vector`), có dùng positive/negative percentile.
  - Tìm anime gần với user vector và loại trừ anime đã biết.

### Đầu vào chính
- Anime metadata + embedding đã train.
- Với user recommendation: `user_ratings` (để dựng sở thích).

### Ưu điểm
- Không phụ thuộc mạnh vào độ dày user-user interactions.
- Tốt cho cold-start item và hỗ trợ giải thích theo nội dung.

### Hạn chế
- Có thể overspecialize theo nội dung gần giống (filter bubble), nên cần diversity re-rank.

### Dùng ở màn nào
- Màn Search (Find Similar Anime): cho phép chọn method `content`.
- Màn Compare: hiển thị cột Content.
- Màn For You (new user hoặc ít dữ liệu): là một thành phần trong tổ hợp.

---

## 3.2 Collaborative (MatrixFactorization wrapper)

### Cách hoạt động
- Trong cấu hình hiện tại, collaborative thường chạy theo nhánh BPR (ranking theo tương tác).
- `recommend_for_user(...)` của collaborative:
  - Với BPR, gọi backend implicit-bpr recommend, có thể loại item đã rated.
- `get_similar_items(...)`:
  - Lấy item tương tự theo latent factors (hoặc gọi similar_items của implicit backend nếu BPR).

### Đầu vào chính
- Ma trận user-item (explicit ratings), mappings user/item.
- Với BPR có thể kết hợp tín hiệu implicit khi train.

### Ưu điểm
- Nắm bắt “hành vi cộng đồng”: người giống nhau thích gì.
- Hữu ích cho existing users có lịch sử đủ dày.

### Hạn chế
- Kém cho cold-start user/item.
- Score cần chuẩn hóa khi phối hợp với mô hình khác.

### Dùng ở màn nào
- Màn Search: chọn method `collaborative` cho anime-to-anime.
- Màn Compare: hiển thị cột Collaborative.
- Màn For You (existing user): là nguồn candidate quan trọng, nhưng thường đứng sau ALS về trọng số.

---

## 3.3 Implicit ALS

### Cách hoạt động
- Mô hình ALS trên implicit feedback (xem, trạng thái xem, episodes watched).
- `recommend_for_user(...)`:
  - Lấy candidate top-K theo user latent factors.
  - Có thể loại known items và bật/tắt diversity ở mức model.
- `get_similar_items(...)`:
  - Tìm item tương tự trong latent space.

### Đầu vào chính
- Implicit matrix đã build từ `animelist.csv`.
- Mapping user/item của training pipeline.

### Ưu điểm
- Rất mạnh cho nhiệm vụ ranking ở existing user.
- Nắm bắt tín hiệu hành vi rộng hơn explicit rating.

### Hạn chế
- Khó giải thích với user không kỹ thuật (latent factors).
- Cần xử lý cold-start bằng mô hình khác.

### Dùng ở màn nào
- Màn Search: cho phép chọn method `implicit`.
- Màn Compare: backend có trả kết quả implicit, nhưng UI hiện tại đang ẩn cột implicit trong HTML.
- Màn For You (existing user): là nguồn retrieval quan trọng nhất (lấy nhiều candidate nhất).

---

## 3.4 Popularity

### Cách hoạt động
- Tính trước các bảng:
  - Top rated
  - Most watched
  - Trending
  - Most members
- `get_popular(...)` chọn danh sách theo loại tab.
- `get_recommendations_for_new_user(...)` trộn điểm từ top-rated/most-watched/trending.

### Đầu vào chính
- `anime_df` + `ratings_df` + `animelist_df` (hoặc bản streaming từ CSV).

### Ưu điểm
- Ổn định, nhanh, rất hợp cold-start.
- Phù hợp các màn khám phá chung, không cần cá nhân hóa.

### Hạn chế
- Cá nhân hóa thấp.
- Có thiên lệch về anime phổ biến.

### Dùng ở màn nào
- Màn Popular: phương thức chính và đúng mục tiêu màn.
- Màn For You (new user): làm xương sống để tránh rỗng dữ liệu.
- Màn Hybrid: đóng vai trò “diversity/popularity boost”.

---

## 3.5 Hybrid / Learned Hybrid

### Cách hoạt động (2 nhóm bài toán)

#### a) Anime -> Anime (`recommend_similar_anime`)
- Stage 1: lấy candidates từ Content (nhiều nhất ở bước đầu).
- Stage 2: mở rộng bằng ALS + Collaborative item similarity.
- Stage 3: kết hợp điểm theo fallback weights (meta-model không áp dụng vì không có user context).
- Stage 4: Genre-aware MMR để tăng đa dạng thể loại.

#### b) User -> Anime (`recommend_for_user`)
- Stage 1 Retrieval:
  - ALS top lớn (nguồn mạnh nhất)
  - Collaborative top trung bình
  - Content top nhỏ hơn
  - Popularity bổ sung candidate
- Stage 2 Re-rank:
  - Nếu meta-model đã train: dùng `predict_proba` trên vector 7 features.
  - Nếu chưa train/lỗi: fallback weighted sum.
- Stage 3 Diversity:
  - Genre-aware MMR (điều chỉnh relevance vs diversity).

### Vì sao hybrid là mặc định
- Mỗi mô hình mạnh ở một vùng khác nhau:
  - ALS mạnh về hành vi ẩn.
  - Collaborative mạnh về pattern cộng đồng.
  - Content mạnh về semantic gần gũi.
  - Popularity giúp ổn định cold-start và tăng độ phủ.
- Cascade + re-rank giúp cân bằng giữa độ chính xác và độ đa dạng.

---

## 4) Phân tích theo từng màn hình web và lý do chọn phương thức

## 4.1 Màn Search (Find Similar Anime)

### API và hành vi
- Frontend gọi: `/api/recommend/anime/{anime_id}?method=...`
- Cho phép chọn trực tiếp `content | collaborative | implicit | hybrid`.

### Vì sao thiết kế như vậy
- Search màn hình là “bài toán item-to-item”, nên cả 4 phương thức đều có ý nghĩa so sánh.
- Cho user/dev thấy rõ khác biệt bản chất giữa các thuật toán.
- `hybrid` đặt mặc định để tối ưu chất lượng tổng thể.

### Giá trị UX
- Vừa phục vụ người dùng cuối (tìm anime tương tự), vừa phục vụ demo kỹ thuật (đổi method để quan sát khác biệt).

---

## 4.2 Màn Popular

### API và hành vi
- Frontend gọi `/api/popular?type=top_rated|most_members|trending`.
- Có filter genre.

### Vì sao dùng popularity cho màn này
- Mục tiêu màn là khám phá xu hướng chung, không phải cá nhân hóa.
- Không cần user context vẫn cho kết quả tốt, nhanh, ổn định.
- Đây là điểm vào phù hợp cho user mới hoặc khách chưa đăng nhập.

---

## 4.3 Màn For You (Personalized)

### API và hành vi
- Frontend gọi `/api/recommend/user/{user_id}`.
- Backend để strategy `auto`:
  - New user -> nhánh cold-start.
  - Existing user -> nhánh personalized đầy đủ.

### Vì sao dùng cơ chế auto-strategy
- Vì dữ liệu user không đồng nhất giữa các user demo.
- Tránh tình trạng user mới nhận kết quả rỗng/kém chất lượng.
- Existing user được tận dụng đầy đủ sức mạnh ALS + collab + content + re-rank.

### Cơ chế theo ngữ cảnh
- New user:
  - Chủ yếu Popularity, cộng thêm Content nếu có tối thiểu vài ratings.
- Existing user:
  - Retrieval đa nguồn, ưu tiên ALS.
  - Meta-model học từ dữ liệu lịch sử để re-rank phù hợp user hơn.
  - MMR để giảm lặp thể loại.

---

## 4.4 Màn Compare

### API và hành vi
- Frontend gọi `/api/compare?anime_id=...`.
- Backend trả kết quả cho cả `content`, `collaborative`, `implicit`, `hybrid`.
- Tuy nhiên HTML hiện tại đang hiển thị 3 cột chính (content, collaborative, hybrid); cột implicit đang được comment.

### Vì sao dùng nhiều phương thức ở màn này
- Mục tiêu chính là minh họa sự khác biệt giữa các thuật toán.
- Đây là màn phù hợp để giải thích cho stakeholder/team vì sao hybrid hợp lý hơn single-model.

---

## 5) Bảng tóm tắt: Màn hình -> Phương thức -> Lý do

| Màn hình | Phương thức chính | Lý do chọn |
|---|---|---|
| Search | Content / Collaborative / Implicit / Hybrid | Bài toán item-to-item, cần cho phép so sánh thuật toán trực tiếp |
| Popular | Popularity | Không cần user context, nhanh, ổn định, đúng mục tiêu khám phá xu hướng |
| For You | Auto (New vs Existing) + Hybrid cascade | Đảm bảo không rỗng cho user mới và tối ưu chất lượng cho user cũ |
| Compare | Content + Collaborative + Hybrid (backend có implicit) | Trình bày khác biệt phương pháp, phục vụ demo và giải thích kỹ thuật |

---

## 6) Vì sao các phương thức này phù hợp với đúng màn (góc nhìn sản phẩm)
- Search cần “độ liên quan theo item đang xem” -> mạnh về similarity theo nội dung/hành vi item.
- Popular cần “danh sách đại diện thị trường” -> popularity là tự nhiên nhất.
- For You cần “cá nhân hóa thực chiến” -> phải dùng nhiều nguồn tín hiệu và re-rank theo user context.
- Compare cần “khả năng giải thích” -> đặt các phương thức cạnh nhau để người xem hiểu trade-off.

---

## 7) Nhận xét kỹ thuật quan trọng
- Web backend gọi `HybridEngine` nhưng thực thi `LearnedHybridEngine` do alias.
- Anime-to-anime hybrid không dùng meta-model (không có user context), chỉ dùng weighted fusion + MMR.
- User-to-anime mới dùng meta-model (nếu đã train); nếu không thì fallback weights.
- Cột implicit ở màn Compare đang ẩn trên UI, dù backend vẫn trả dữ liệu.

---

## 8) Kết luận
Thiết kế hiện tại là hợp lý theo từng màn:
- Màn tổng quan/trending dùng Popularity để đảm bảo nhanh và ổn định.
- Màn cá nhân hóa dùng Hybrid cascade để tối đa hóa chất lượng.
- Màn so sánh cho phép quan sát khác biệt và giải thích quyết định kỹ thuật.

Nếu cần nâng cấp bước tiếp theo, nên ưu tiên:
1. Đồng bộ hiển thị Compare với dữ liệu backend (quyết định rõ có hiển thị implicit hay không).
2. Bổ sung chỉ số online/offline theo từng màn để đo tác động của từng phương thức.
3. Chuẩn hóa tài liệu "method selection policy" để team frontend/backend dùng chung tiêu chí.

