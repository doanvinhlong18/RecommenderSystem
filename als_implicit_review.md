# Implicit Pipeline Review (ALS)

## 1) Phạm vi review
- Mục tiêu: mô tả đầy đủ pipeline implicit recommendation trong dự án, bao gồm **dataset sử dụng**, **input**, **tiền xử lý/build matrix**, **quá trình train/infer**, và **output artifact**.
- Nguồn code đối chiếu chính:
  - `preprocessing/data_loader.py`
  - `preprocessing/data_splitter.py`
  - `preprocessing/matrix_builder.py`
  - `models/implicit/als_implicit.py`
  - `train.py`
  - `config.py`

## 2) Dataset dùng cho implicit là gì?
Implicit model không học trực tiếp từ điểm `rating` mà học từ hành vi xem trong `animelist.csv`.

### File liên quan
- `data/animelist.csv`
  - Cột dùng cho implicit:
    - `user_id`
    - `anime_id`
    - `watching_status`
    - `watched_episodes`
- `data/rating_complete.csv`
  - Không phải nguồn score implicit, nhưng dùng để tạo split train/test và mapping user/item gốc cho training pipeline.
- `data/watching_status.csv`
  - Có load trong `DataLoader`, nhưng trong `MatrixBuilder.build_implicit_matrix*` hiện tại chưa sử dụng bảng map này; weights đang hard-code.

### Cấu hình dataset
- Định nghĩa ở `config.py`:
  - `data_config.animelist_file = "animelist.csv"`
  - `data_config.rating_file = "rating_complete.csv"`
- `DataLoader` đọc file theo chunk + dtype tối ưu (`int32`, `int8`) để giảm RAM.

## 3) Input của implicit model
Input trực tiếp cho `ALSImplicit.fit(...)`:
- `implicit_matrix: csr_matrix` kích thước `[n_users x n_items]`
- `anime_to_idx`, `idx_to_anime`
- `user_to_idx`, `idx_to_user`
- Tùy chọn:
  - `negative_matrix` (phạt explicit negatives)
  - `interaction_dates` (để time-decay)

Trong luồng train chuẩn (`train.py`), `negative_matrix` và `interaction_dates` chưa được cấp vào, nên mặc định pipeline đang sử dụng:
- implicit score cơ bản từ `animelist`
- không time-decay
- không negative sampling bổ sung (ngoài score thấp từ status).

## 4) Tiền xử lý trước khi build implicit matrix

### 4.1 Tạo train/test split và chống leakage
Có 2 chế độ:
1. **Sample mode** (`--sample-size`):
   - Tạo split in-memory từ ratings (`create_ratings_user_split`).
   - Loại bỏ held-out user-item khỏi animelist bằng `filter_holdout_interactions`.
2. **Full-data mode**:
   - Tạo split trên đĩa bằng `create_ratings_disk_split` (2-pass stream).
   - Tạo `train_animelist.csv` đã lọc holdout bằng `filter_animelist_to_disk`.

=> Mục tiêu: cặp user-item dành cho evaluation không bị lộ vào train implicit.

### 4.2 Build mapping explicit trước
`MatrixBuilder.build_rating_matrix*` được gọi trước implicit để tạo:
- `user_to_idx`, `idx_to_user`
- `anime_to_idx`, `idx_to_anime` (explicit universe)

### 4.3 Build implicit matrix từ animelist
Hàm:
- `build_implicit_matrix(...)` (DataFrame)
- `build_implicit_matrix_from_csv(...)` (stream CSV)

Các bước logic:
1. Chỉ giữ interactions của tập user trong `user_to_idx` (user train).
2. Đếm tần suất anime trong implicit data.
3. Mở rộng item universe:
   - `explicit_anime_set` U `implicit_only`
   - `implicit_only` là anime có số lần xuất hiện >= `min_implicit_anime_interactions` (mặc định 50) và không có trong explicit ratings.
4. Tạo mapping riêng cho implicit:
   - `implicit_anime_to_idx`
   - `implicit_idx_to_anime`
5. Tính implicit score mỗi dòng theo công thức:

```text
status_w = map(watching_status)
episode_w = clip(log1p(watched_episodes) / 10, 0, 1)
score    = 0.6 * status_w + 0.4 * episode_w
```

Status weights hard-code hiện tại:
- 1 (watching): 0.8
- 2 (completed): 1.0
- 3 (on-hold): 0.5
- 4 (dropped): 0.2
- 6 (plan-to-watch): 0.1
- status khác/NaN: 0.1

6. Build `csr_matrix(scores, (row_idx, col_idx), shape=(n_users, n_implicit_items))`.

## 5) Process train trong `ALSImplicit`

### 5.1 Khởi tạo
`ALSImplicit.__init__` lấy hyperparams từ `model_config`:
- `implicit_factors`
- `implicit_iterations`
- `implicit_regularization`
Thêm:
- `alpha` (scale confidence)
- `negative_weight`
- `use_temporal_weighting`
- `diversity_lambda`

### 5.2 fit pipeline
Trong `fit(...)`:
1. Lưu mappings.
2. Nếu bật `use_temporal_weighting` + có `interaction_dates`:
   - Nhân decay theo half-life.
3. Nếu có `negative_matrix`:
   - Trừ confidence cho explicit negatives.
4. Train bằng thư viện `implicit` (ưu tiên GPU nếu có), nếu không có library thì fallback custom ALS.

### 5.3 Confidence transform trước khi fit
`_fit_with_implicit_library`:
- Chuyển matrix thành `item_user_matrix = matrix.T`
- Scale confidence:

```text
c_ui = 1 + alpha * score_ui
```

- Gọi `self._implicit_model.fit(item_user_matrix)`.

### 5.4 Factor handling
Sau train:
- Lấy `user_factors`, `item_factors`
- Có `_resolve_factors()` để xử lý trường hợp orientation bị đảo.

## 6) Output của implicit model

### 6.1 Output infer
- `recommend_for_user(...)` trả list:
  - `[{"mal_id": int, "score": float}, ...]`
- `get_similar_items(...)` trả list:
  - `[{"mal_id": int, "similarity": float}, ...]`

Nếu user không có trong train mappings:
- fallback sang `_get_popular_items` (dựa trên L2 norm của item factors).

### 6.2 Output artifact trên đĩa
- Save model qua `ALSImplicit.save(filepath)` (pickle state):
  - hyperparams + factors + mappings
- Trong pipeline train tổng:
  - matrix artifacts ở `saved_models/matrices/`:
    - `implicit_matrix.npz`
    - `user_item_matrix.npz`
    - `item_user_matrix.npz`
    - `mappings.pkl`
  - model implicit được đóng gói trong learned hybrid save dir (`saved_models/learned_hybrid/`).

## 7) End-to-end flow (tóm tắt)
1. Đọc `rating_complete.csv` + `animelist.csv`.
2. Tạo split train/test từ ratings.
3. Lọc holdout pairs khỏi animelist train để tránh leakage.
4. Build explicit rating matrix -> có user/item mappings.
5. Build implicit matrix từ animelist train (score theo status + episodes, mở rộng implicit items).
6. Train `ALSImplicit.fit(...)` trên implicit matrix.
7. Serve recommendation qua `recommend_for_user` / `get_similar_items`.

## 8) Các điểm cần lưu ý khi vận hành
- `watching_status_df` đang chưa được sử dụng trong matrix scoring (weights hard-code trong code).
- Pipeline train mặc định chưa truyền `negative_matrix` và `interaction_dates`; các tính năng này tồn tại trong class nhưng chưa active ở train flow chuẩn.
- Implicit item universe có thể lớn hơn explicit universe (có thêm `implicit_only` item), đây là chủ đích để bao phủ anime có hành vi xem cao nhưng ít rating.
- `recommend_for_user` với thư viện implicit đang gọi `filter_already_liked_items=False` và lọc tay theo `known_items`; cần đảm bảo caller truyền `known_items` đúng để tránh gợi lại item đã biết.

## 9) Input/Process/Output checklist (để đối chiếu nhanh)
- Input data:
  - `animelist.csv` (hành vi xem) + mappings user/item từ ratings split.
- Process:
  - split -> remove holdout -> build mappings -> build implicit scores -> ALS confidence train.
- Output:
  - top-K recommendation/similar items + model/matrix artifacts để phục vụ API/hybrid.
