# Duyệt dataset và chuẩn bị baseline

## Đã triển khai

- `tools/review_dataset.py generate`: xác minh dữ liệu nguồn, quét perceptual hash
  và ArcFace đã căn chỉnh 5 điểm, xuất ảnh xem nhanh và CSV quyết định.
- Phát hiện nhiều mặt, mặt nhỏ hoặc không có mặt thành các mục cần duyệt riêng.
  Từng ảnh đều có mục review; điểm tương đồng không tự xác nhận danh tính.
- Quét lưu từng đặc trưng vào file riêng theo cách atomic. `--resume` dùng lại
  phần đã xong khi dataset/config/model/code vẫn khớp. Không ghi đè review hoàn tất.
- `tools/dataset_review_app.py`: giao diện cục bộ, lọc mục chờ/cặp khác tập,
  xem ảnh cạnh nhau và lưu từng quyết định có tên người duyệt.
- `freeze`: từ chối mục pending, dữ liệu thay đổi, quyết định mâu thuẫn, rò rỉ
  pixel và loại ảnh holdout. Nhóm cùng người được xử lý bắc cầu; giữ holdout trước,
  sau đó validation; loại thành viên train khỏi manifest mới, không xóa ảnh.
- `prepare_baseline.py`: chỉ nhận split frozen có review receipt; rehash dữ liệu,
  kiểm tra benchmark đúng tập holdout, ghi run manifest và chạy full khi có `--execute`.
- Training production dùng cùng review gate; sửa riêng chuỗi `approved` không đủ.

## Sử dụng trên máy này

Lượt quét đầu bị ngắt trước khi tạo report; thư mục đó được giữ nguyên.
Lượt có checkpoint tiến độ dùng `benchmarks/reports/dataset_review_v2`:

```powershell
.venv/Scripts/python.exe -B -u tools/review_dataset.py generate --output benchmarks/reports/dataset_review_v2 --resume
.venv/Scripts/python.exe -m streamlit run tools/dataset_review_app.py --server.address 127.0.0.1 --server.port 8502
```

Chỉ dùng lệnh resume nếu scan bị ngắt; không chạy thêm một tiến trình trên cùng
thư mục khi scan vẫn đang hoạt động. Giao diện ở http://127.0.0.1:8502; dùng nút
tải lại tiến độ. Khi report hoàn tất, tải lại trang để bắt đầu duyệt. File HTML
`index.html` là lựa chọn xem offline; `decisions.csv` là dữ liệu quyết định chung.

Sau khi tất cả mục đã được người duyệt quyết định:

```powershell
.venv/Scripts/python.exe -B tools/review_dataset.py freeze --review-dir benchmarks/reports/dataset_review_v2 --reviewer "TEN_NGUOI_DUYET" --output benchmarks/splits/real_portraits_reviewed_v1
.venv/Scripts/python.exe -B tools/prepare_baseline.py --split-dir benchmarks/splits/real_portraits_reviewed_v1 --output benchmarks/reports/baseline_reviewed_v1 --execute
```

Giữ mục không chắc chắn ở pending. Không dùng `distinct` chỉ để bỏ qua cặp khó.
Nếu cần loại ảnh holdout, phải tạo benchmark version mới và baseline tương ứng.
Nếu loại ảnh khiến train <1.000 hoặc validation <20 thì cần bổ sung dữ liệu trước.

## Kiểm chứng

- Master regression: 10/10 suite, exit 0 (`review_regression_final.log`).
- Review unit/UI: 11/11 test, exit 0 (`review_ui_tests.log`), gồm lưu quyết định qua
  Streamlit AppTest, tiếp tục scan bị ngắt, xử lý nhóm bắc cầu, chặn holdout exclusion
  và chặn baseline khi chỉ sửa trạng thái approved.
- Kiểm tra trực tiếp split thật đang pending: baseline bị chặn trước khi nạp model
  và trước khi tạo thư mục output.
- Giao diện tiến độ trên scan thật đã render bằng AppTest không có exception.

Chưa hoàn tất review thủ công, chưa đóng băng split, chưa chạy baseline đầy đủ,
chưa khởi chạy GPU training và chưa đổi model production. Ảnh, embedding và quyết
định riêng tư nằm trong thư mục Git/Docker ignored; chỉ công cụ/tài liệu được commit.
