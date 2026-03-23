# ALS Implicit Module Review

## Tổng quan
- File: `models/implicit/als_implicit.py` triển khai ALS cho dữ liệu implicit feedback với khả năng dùng GPU thông qua thư viện `implicit` và có fallback sang solver tự cài đặt. Module xử lý mapping ID, huấn luyện, recommend, tìm item tương tự và lưu/đọc model.

## Giải thích theo từng phần
- **Khởi tạo class** (`ALSImplicit.__init__` tại `als_implicit.py:37`): đọc cấu hình mặc định từ `model_config`, xác định device qua `get_device`, thiết lập cờ GPU, khởi tạo ma trận factor và mapping ID. Logging được cấu hình ngay khi import module.
- **fit** (`als_implicit.py:79`): lưu mapping ID, log kích thước ma trận, quyết định có dùng GPU hay không, log bộ nhớ GPU, thử huấn luyện bằng thư viện trước rồi fallback sang custom ALS nếu cần. Ghi lại thời gian chạy. Yêu cầu input mapping không rỗng.
- **Huấn luyện bằng thư viện** (`_fit_with_implicit_library` tại `als_implicit.py:131`): chọn class ALS qua `get_implicit_als_class`, cấu hình số factor/số iteration/regularization. Chuyển ma trận sang item-user CSR, scale dữ liệu với `1 + alpha * r`, fit model, sau đó chuyển dữ liệu GPU về numpy bằng `.to_numpy` nếu cần.
- **Huấn luyện custom** (`_fit_custom` tại `als_implicit.py:175`): triển khai theo công thức Hu et al. Khởi tạo ngẫu nhiên, cập nhật luân phiên theo user và item với ma trận confidence. Dùng giải hệ phương trình dạng dense trong vòng lặp Python → chậm O(users × items) với dữ liệu lớn; không có early stopping hoặc kiểm tra hội tụ.
- **Recommendation** (`recommend_for_user` tại `als_implicit.py:243`): xác định index user, xử lý hoán đổi factor nếu thư viện lưu ngược. Nhánh thư viện gọi `self._implicit_model.recommend` nhưng không truyền `filter_items`, nên việc loại item đã biết chỉ làm thủ công sau đó qua `known_items`. Nhánh custom tính score bằng dot product; item đã biết gán `-inf`. Có dòng `return results` bị lặp (code chết).
- **Item tương tự** (`get_similar_items` tại `als_implicit.py:357`): ưu tiên dùng `similar_items` của thư viện, nếu không thì dùng cosine similarity trên item factors với chuẩn hóa đơn giản. Không có re-scale score hoặc điều chỉnh theo popularity.
- **Lưu/đọc model** (`save`/`load` tại `als_implicit.py:426`): dùng pickle để lưu factors, hyperparameters và mapping ID.
- **Main block** (`als_implicit.py:471`): demo load data, build matrix, train và test recommend/similar; hữu ích để test nhanh nhưng chưa tách bằng argparse/config.

## Vấn đề và rủi ro
- `filter_items` được chuẩn bị nhưng không truyền vào `recommend`; việc loại item đã biết phụ thuộc vào xử lý sau. Nếu thư viện implicit dùng cache `user_items`, có thể gây leakage.
- Logic hoán đổi shape của factor có thể che giấu lỗi mapping upstream; nên xử lý rõ ràng hơn (hoặc lưu flag).
- Vòng lặp custom ALS rất tốn tài nguyên và có thể không ổn định với user/item không có tương tác (không kiểm tra user rỗng khi update).
- Dòng `return results` bị lặp trong `recommend_for_user` là dư thừa; tồn tại code không bao giờ chạy.
- Logging dùng `basicConfig` global khi import; dễ gây xung đột với hệ thống logging của ứng dụng.

## Đề xuất cải thiện chất lượng recommend
1. **Tinh chỉnh confidence**: trước ALS, áp dụng weighting (BM25/TF-IDF hoặc log-scaling) cho implicit data, và grid-search `alpha`, `regularization`, `factors`, `iterations` bằng các metric offline (MAP@K, NDCG@K).
2. **Lọc item tốt hơn**: truyền `user_items`/`filter_items` trực tiếp vào `implicit.recommend` để loại item đã biết ngay từ bước top-K.
3. **Xử lý cold-start**: bổ sung fallback theo popularity hoặc content-based; có thể warm-start item mới bằng embedding nội dung.
4. **Chuẩn hóa & scoring**: L2-normalize factor sau train nếu dùng cosine; có thể calibrate score (sigmoid) nếu cần output bounded.
5. **Cập nhật theo thời gian**: áp dụng time decay hoặc tăng trọng số cho tương tác gần đây.
6. **Tăng tốc huấn luyện**: ưu tiên dùng solver của thư viện (CPU/GPU); nếu dùng custom thì vectorize và thêm early stopping.
7. **ANN cho similar items**: dùng FAISS hoặc HNSW để tăng tốc truy vấn item tương tự.
8. **Bias & diversity**: thêm bước re-ranking để tăng diversity/novelty, hoặc giới hạn ảnh hưởng của item quá phổ biến.
9. **Pipeline đánh giá**: xây dựng script đánh giá offline để so sánh ALS với các model khác (BPR, VAE…).

## Sửa nhanh (ít effort)
- Truyền `filter_items` vào `self._implicit_model.recommend` và xoá dòng `return` dư.
- Thêm kiểm tra user không có tương tác trong custom ALS.
- Di chuyển cấu hình logging ra ngoài hoặc bọc trong `if __name__ == "__main__":`.