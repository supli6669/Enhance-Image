# Kế hoạch cải thiện Custom AI Enhancer — 2026-09-08

Trạng thái: kế hoạch đề xuất cho đợt tiếp theo; chưa khởi chạy training.
Người dùng đã yêu cầu triển khai. Đợt P1/P2: công cụ review và đóng băng split
đang được triển khai; chưa hoàn thành review thủ công hoặc baseline đầy đủ.
Phạm vi: phục hồi ảnh người thật, giữ nhận dạng, chạy ổn định trên CPU.
Kế hoạch recovery trong IMPLEMENTATION_PLAN.md đã hoàn thành; tài liệu này nối
tiếp phần benchmark và review dữ liệu còn thiếu, trước khi cải thiện model.

## Điểm xuất phát

- Mã nguồn recovery: f7c0974; 9/9 suite local và CPU CI GitHub đã pass.
- Bộ dữ liệu hiện có: 2.479 train, 130 validation, 391 ảnh benchmark duy nhất.
  Đây là số trước review danh tính; số cuối có thể giảm sau xử lý rò rỉ dữ liệu.
- Đã loại trùng pixel; chưa duyệt hết ảnh gần giống và cùng một người giữa các tập.
- Baseline gốc vẫn là mặc định. Smoke 3 ảnh không đủ kết luận chất lượng v3.
- Local có baseline ONNX đã kiểm tra parity. Docker vẫn dùng baseline PyTorch.
- GitHub đã nhận code; lần đồng bộ HF gần nhất thất bại, push local báo lỗi xác thực.
- Kaggle key cũ cần thu hồi. Không tiếp tục checkpoint toy 6.000 iteration.

## Mục tiêu và giới hạn

1. Đo được mức cải thiện trên dữ liệu độc lập, tránh làm mặt sắc hơn nhưng sai người.
2. Có training tái lập được, checkpoint có nguồn gốc và khả năng khôi phục khi ngắt phiên.
3. Giữ CPU làm đích triển khai: detector nhẹ, Lanczos mặc định, không tự bật neural face upscale.
4. Giữ cùng tham số xử lý cho ảnh/video/batch; báo lỗi và model đang dùng rõ ràng.
5. Không mở rộng anime/game, thêm kiến trúc mới hay tăng dataset hàng loạt ở vòng đầu.
6. Không tự xóa ảnh, ghi đè checkpoint, đổi model production hoặc đánh dấu split approved.

## Các giai đoạn theo thứ tự

| Giai đoạn | Công việc | Đầu ra và tiêu chí hoàn thành |
| --- | --- | --- |
| P0 — Khôi phục vận hành | Thu hồi key Kaggle cũ; cấu hình credentials riêng tư; sửa HF authentication; kiểm tra nguồn dataset và quyền sử dụng | Không còn dùng key cũ; Kaggle đọc được dataset riêng tư; HF nhận đúng commit và health check thành công. HF không chặn chuẩn bị dữ liệu/training nhưng chặn phát hành |
| P1 — Duyệt dữ liệu | Tạo danh sách ảnh gần giống, ảnh lỗi, mặt quá nhỏ và nhóm có thể cùng người; tạo contact sheet để duyệt; chia theo người trước khi tạo biến thể degradation | Split mới có version/hash, zero trùng pixel, không còn cặp rò rỉ đã xác nhận; reviewer ghi quyết định. Nhóm chưa rõ để riêng; không xóa file gốc |
| P2 — Đóng băng baseline | Sau P1, cố định benchmark; chạy đủ PSNR/SSIM/LPIPS/ArcFace, số mẫu đo được và kết quả từng nhóm; đo CPU riêng với warm-up | Báo cáo đầy đủ cùng model/data/config hash. Coverage ArcFace và lỗi xử lý được công bố; có median/p95 latency, peak RAM và bộ ảnh A/B cố định |
| P3 — Chuẩn bị Kaggle | Đóng gói dataset/split/weights/baseline riêng tư; xác minh teacher; chạy preflight và 2-iteration verify trên GPU; thử lưu/nạp checkpoint của run mới | Loss/gradient hữu hạn, teacher frozen đúng, validation hoạt động, resume khớp iteration/scheduler; ghi GPU, VRAM, giây/iteration và dự toán thời gian trước run dài |
| P4 — Train Stage III fresh | Dùng cấu hình hiện có và pretrained khởi đầu; chạy một pilot có validation định kỳ; lưu checkpoint riêng cho mỗi run | Chọn checkpoint theo validation và review ảnh; không chọn theo train loss. Không NaN/OOM; lưu đủ config, seed, data hash, code commit, logs và checkpoint hashes |
| P5 — Đánh giá và quyết định | So candidate đã chọn với baseline trên cùng benchmark; review A/B ẩn tên model; phân tích nhóm thất bại | Báo cáo quality/identity/latency và quyết định giữ baseline hoặc đưa candidate sang tối ưu; không tự promote |
| P6 — Tối ưu CPU và phát hành | Export FP32 kiểm tra parity; INT8 dùng crop train để calibration; đánh giá lại; sửa packaging HF; smoke ảnh/video/batch trên Space | Không dùng holdout calibration; graph/sidecar đủ hash; đạt tiêu chí chất lượng và tài nguyên; health check pass, có manifest release và cách rollback baseline |

P1–P2 là việc nên làm ngay. P3–P4 chỉ bắt đầu khi split được duyệt và baseline đầy
đủ. Không chuyển Stage II trước khi hoàn thành và đánh giá Stage III; chỉ xem xét
Stage II ở vòng sau nếu bằng chứng cho thấy giới hạn nằm ở transformer.

## Thiết kế đầu ra để tái lập

- Tái sử dụng prepare_training_split.py, training_preflight.py, evaluator,
  compare_evaluations.py, export_onnx.py và quantize_onnx_static.py.
- Bổ sung công cụ review tạo `review_candidates.csv` gồm đường dẫn, hash, nhóm
  nghi trùng, lý do, quyết định và reviewer; điểm tương đồng chỉ gợi ý, không tự
  xác nhận danh tính. Contact sheet và ảnh riêng tư nằm trong thư mục ignored.
- Split có train/validation/holdout manifests, nguồn dữ liệu, quyết định review
  và hash. Trạng thái: draft → pending_review → approved → frozen. Đổi ảnh sau
  frozen phải tạo version mới và chạy lại baseline tương ứng.
- Run có run_id duy nhất và manifest liên kết code commit, split hash, pretrained
  hashes, config, seed, môi trường, checkpoint và báo cáo. Resume chỉ từ cùng
  run/config tương thích; không nhận checkpoint toy lịch sử.
- Model: staged → validated → reviewed → released. Export thành công không đồng
  nghĩa đủ chất lượng. Model bị loại vẫn được giữ cùng lý do, không ghi đè bản tốt.

## Tiêu chí chọn candidate đề xuất

Chốt các ngưỡng trước khi xem candidate trên benchmark. Các số dưới đây là mục
tiêu kỹ thuật ban đầu, không phải chất lượng đã đạt hay bảo đảm nhận dạng:

- Median LPIPS giảm ít nhất 5% so baseline; median ArcFace không giảm quá 0,005.
- Median PSNR không giảm quá 0,2 dB và SSIM không giảm quá 0,005. Đọc cả kết quả
  theo nhóm, không để trung bình che mất lỗi ở mặt nghiêng/kính/ánh sáng khó.
- Review A/B ít nhất 50 ảnh đa dạng, bao gồm nhóm rủi ro; không còn lỗi đổi nét
  nhận dạng nghiêm trọng đã xác nhận. Không dùng kết quả test để chỉnh từng vòng.
- Lượt xử lý lỗi phải là 0 trên benchmark hợp lệ; coverage identity của candidate
  không giảm so baseline. Ảnh không đo được ArcFace phải xuất hiện trong báo cáo.
- CPU median và p95 latency không tăng quá 10% so baseline trên cùng máy, cùng
  thread/input/model settings, không chạy tác vụ nặng khác. Peak RAM phải nằm
  trong giới hạn thực của Space, có khoảng trống cho ảnh lớn và nhiều khuôn mặt.
- INT8 được so riêng với FP32 candidate; parity FP32 và quality INT8 là hai phép
  kiểm tra khác nhau. Không đạt thì giữ FP32 hoặc baseline, không ép phát hành.

Các ngưỡng cần được kiểm tra tính phù hợp với độ biến động của baseline. Nếu cần
đổi, ghi lý do và chốt trước lần đánh giá candidate kế tiếp, không nới ngưỡng chỉ
để hợp thức hóa một model đã thất bại. Validation dùng để điều chỉnh; benchmark
đóng băng dành cho quyết định cuối, tránh tối ưu lặp lại theo test.

## Khi nào tăng dataset?

Chưa đặt mục tiêu số lượng tùy ý. Dùng bộ hiện tại làm pilot sau review. Khi
validation cho thấy lỗi tập trung ở nhóm nào, bổ sung ảnh chất lượng tốt đúng
nhóm đó: mặt nghiêng, kính, ánh sáng khó, nhóm tuổi hoặc khuôn mặt còn thiếu.
Mỗi lần bổ sung phải kiểm tra quyền dùng, trùng ảnh và rò rỉ theo người; giữ holdout
độc lập. So với cùng baseline và ghi riêng ảnh hưởng của dữ liệu/loss/degradation,
không thay tất cả cùng một lần. GT tránh ảnh đã làm đẹp mạnh hoặc upscale bằng AI.

## Phân công và điểm bắt đầu

- Người dùng: thu hồi key, cấu hình secrets riêng tư, chuẩn bị GPU/quota Kaggle,
  xác nhận quyền dùng ảnh và duyệt các cặp/nhóm còn nghi ngờ qua contact sheet.
- Công việc kỹ thuật: tạo bộ review, xác minh split, chạy baseline, đóng gói run,
  kiểm thử GPU, triển khai training đã đủ điều kiện, đánh giá và tối ưu CPU.
- Ưu tiên đợt đầu: P1 + P2; P0 xử lý phần tài khoản song song. Chỉ dự toán run GPU
  sau phép đo P3, không hứa ngày train xong dựa trên smoke CPU.
- Checklist: [ ] P0; [ ] P1; [ ] P2; [ ] P3; [ ] P4; [ ] P5; [ ] P6.

Trong phiên lập kế hoạch này không khởi chạy benchmark dài, GPU training, tải
dataset lên dịch vụ hoặc thay model mặc định.
