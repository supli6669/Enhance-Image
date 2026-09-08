# Kiểm thử GPU Kaggle trước pilot

Gói riêng này chỉ kiểm thử 2 iteration. Không dùng notebook training cũ để chạy
toàn bộ các cell: notebook cũ gọi full training, không phù hợp gate thử nghiệm.

## Dữ liệu và baseline

- Split thử nghiệm: 2.427 train, 122 validation; giữ 500 đường dẫn holdout ứng với
  391 ảnh có pixel duy nhất. Không chứng nhận tách theo người.
- Baseline đã hoàn tất 391/391, exit 0: mean PSNR 27,4403 dB, SSIM 0,6874,
  LPIPS 0,3762; median latency 7,40992 giây, p95 14,422145 giây.
- Gói bao gồm đúng ảnh trong các manifest, split, quality report, baseline report,
  pretrained CodeFormer/VQGAN/ArcFace và mã nguồn hiện tại. Không có lịch sử Git,
  checkpoint toy, dữ liệu ngoài manifest hoặc credentials.

## File chuẩn bị

Thư mục `artifacts/kaggle_verify_bundle_v1/` chứa:

- `kaggle_verify_payload.zip`: payload riêng tư.
- `bundle_manifest.json`: danh sách file, kích thước và SHA-256.
- `gpu_verify.ipynb`: notebook chạy kiểm thử, không có output hoặc secrets.

Gói hiện tại: 3.222 file, payload 2.312.194.347 byte (~2,31 GB).
Đã giải nén sang thư mục kiểm tra mới, xác minh toàn bộ file hash và chạy preflight
từ mã nguồn/dữ liệu/weights trong gói: exit 0, đúng 2.427/122 và 391 holdout duy nhất.
Master regression 12/12 suite pass; guard `--require-gpu` đã được kiểm tra là dừng
trên CPU. Chưa có bằng chứng job GPU thực thi thành công.

Để tạo phiên bản mới, dùng thư mục mới:

```powershell
python tools/kaggle_verify_bundle.py --output artifacts/kaggle_verify_bundle_v2
```

Mã nguồn được lấy từ working tree và hash từng file; manifest ghi cả commit gốc
và việc có thay đổi source chưa commit lúc đóng gói. Hash file là nguồn xác minh
chính xác nội dung bundle, không chỉ tên commit.

## Chạy trên Kaggle

1. Tạo một **private dataset**, tải `kaggle_verify_payload.zip` và
   `bundle_manifest.json` lên cùng thư mục.
2. Import `gpu_verify.ipynb` thành notebook riêng tư. Attach dataset vừa tạo.
3. Bật GPU accelerator và Internet. Notebook giữ torch/torchvision CUDA sẵn có,
   cài dependency còn thiếu trong môi trường Kaggle, rồi gọi preflight ở process mới.
4. Run all. Notebook dừng nếu không có CUDA, hash sai, thiếu dữ liệu hoặc lệnh lỗi.
   Mỗi lần tạo thư mục working mới; không xóa checkout/checkpoint của lần trước.
5. Lưu output `enhancer_verify_*/gpu_verify_report.json` và `gpu_verify.log`.
   Checkpoint trong `code/models/CodeFormer/experiments/CodeFormer_gpu_verify_*/`.

## Kiểm tra thực hiện trong job

- Training thật trên 2 iteration, batch 1; kiểm tra loss và gradient hữu hạn.
- Validation dùng 2 ảnh thật trong validation manifest, chỉ để kiểm tra luồng chạy.
- Lưu generator, discriminator và state ở iteration 2; đọc lại optimizer/scheduler
  state và ghi hashes. Đây chưa phải kiểm chứng resume tiếp tục thực thi.
- Không chạy pilot dài, export, INT8 hoặc thay model mặc định.

Trạng thái `gpu_two_iteration_check_passed` chỉ có sau khi job GPU thật hoàn tất.
Kiểm thử CPU của công cụ đóng gói không thay thế kết quả đó. Nếu job lỗi, đọc log
và sửa nguyên nhân trước khi khởi chạy pilot. Bước kế tiếp sau khi GPU pass là
kiểm chứng resume thực thi và chuẩn bị cấu hình pilot có giới hạn riêng.

## Xác thực

Máy hiện chưa có credentials Kaggle sử dụng được: các biến môi trường tương ứng
không có và file `.kaggle/kaggle.json` không phải JSON hợp lệ. Chưa gửi job lên cloud.
Thu hồi key từng bị lộ và cấu hình key mới riêng tư, hoặc dùng giao diện Kaggle để
tải gói và chạy notebook. Không gửi key vào chat và không đưa nó vào dataset.
