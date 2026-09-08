# Duyệt dataset và chuẩn bị baseline

## Luồng mặc định mới: duyệt nhanh cho thử nghiệm

Người dùng đã yêu cầu giảm khối lượng duyệt thủ công. Không cần hoàn thành 3.663
quyết định để dùng luồng thử nghiệm này. Giữ luồng review đầy đủ phía dưới làm
lựa chọn riêng; không tự điền quyết định thay người dùng.

Đã tạo `benchmarks/splits/portraits_experiment_v1`: 2.427 train, 122 validation,
giữ nguyên 500 đường dẫn holdout (391 ảnh có pixel duy nhất). Tạm loại 60 ảnh
(52 train, 8 validation) liên quan tới các cặp khác tập đã được gợi ý. Đây là loại
bảo thủ khỏi manifest, không phải kết luận cùng người và không xóa ảnh gốc.

Kiểm tra kỹ thuật trên 3.109 đường dẫn gồm khả năng đọc ảnh, hash, kích thước,
trùng pixel, độ sắc nét toàn ảnh và vùng sáng/tối cực trị. 93 ảnh còn lại có cờ
heuristic, không tự bị loại vì cờ đó. Giao diện hiển thị 40 ảnh mẫu cố định và tối
đa 12 ngoại lệ để xem tùy chọn, không bắt bấm duyệt từng ảnh. Danh sách đầy đủ nằm
trong `quality.json`; danh sách ảnh tạm loại nằm trong `split.json`.

Tại localhost:8502, chọn **Duyệt nhanh — split thử nghiệm**. Split này có trạng thái
`experimental`, không được chứng nhận tách theo người. Ảnh trùng người chưa được
gợi ý vẫn có thể tồn tại; không dùng kết quả thử nghiệm để tự promote production.

```powershell
python tools/prepare_experimental_split.py --output benchmarks/splits/portraits_experiment_v1
python tools/prepare_baseline.py --experimental --split-dir benchmarks/splits/portraits_experiment_v1 --model weights/CodeFormer/codeformer_baseline.onnx --output benchmarks/reports/experiment_baseline_run_v1 --execute
```

Lệnh tạo split từ chối ghi đè thư mục đã có. Đã chuẩn bị run manifest riêng tại
`benchmarks/reports/experiment_baseline_prepared_v1` (không chạy inference đầy đủ).
Khi chạy thật hãy dùng một output mới như ví dụ. Báo cáo mang nhãn experimental;
production training/promotion gate vẫn từ chối nó. Baseline thử nghiệm không yêu
cầu phê duyệt thủ công toàn bộ danh sách cũ.

Kiểm chứng bổ sung: 6 kiểm thử offline cho quarantine/gate/UI và 11 kiểm thử
review cũ pass. AppTest trên dữ liệu thật hiển thị 2.427/122/60 và không có nút
quyết định bắt buộc. Holdout byte-identical; tất cả ảnh tạm loại còn trên đĩa.
Master regression cuối: 11/11 suite pass, exit 0 (`experiment_regression.log`).
Chuẩn bị baseline thử nghiệm exit 0 (`experiment_baseline_prepare.log`), chưa chạy
inference toàn benchmark.

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

Lượt này đã hoàn tất, exit 0: 3.109 đường dẫn ảnh (gồm 500 đường dẫn holdout,
chưa gộp 391 ảnh holdout có pixel duy nhất), 3.663 mục và 62 trang HTML.
Trong đó có 119 ảnh được detector gắn cờ nhiều mặt, 140 gợi ý ảnh gần trùng,
414 gợi ý cùng người; 91 cặp gợi ý cùng người nằm ở hai tập khác nhau.
Đây là gợi ý chưa xác nhận; cả 3.663 mục vẫn pending. Nên bật bộ lọc cặp khác
tập để duyệt 91 cặp đó trước, sau đó hoàn tất những mục còn lại.

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
- Giao diện trên toàn bộ report thật và bộ lọc cặp khác tập đã qua AppTest;
  không thay đổi quyết định thật trong quá trình kiểm thử.
- CPU CI GitHub đã pass; HF tự động đồng bộ đúng commit 26efe9c và Space báo
  RUNNING. Push trực tiếp từ máy vẫn chưa có credential, nhưng đường CI hoạt động.

Chưa hoàn tất review thủ công, chưa đóng băng split, chưa chạy baseline đầy đủ,
chưa khởi chạy GPU training và chưa đổi model production. Ảnh, embedding và quyết
định riêng tư nằm trong thư mục Git/Docker ignored; chỉ công cụ/tài liệu được commit.
