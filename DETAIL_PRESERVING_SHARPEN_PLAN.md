# Kế hoạch làm nét rõ hơn, giữ cấu trúc ảnh

Ngày: 2026-09-13. Trạng thái: kế hoạch, chưa triển khai các pha bên dưới.

## 1. Mục tiêu và phạm vi

Mục tiêu sản phẩm: ảnh mờ nhẹ/vừa nhìn rõ hơn ở cùng kích thước hiển thị, giữ
đường nét mặt, hình dạng đồ vật, chữ và màu sắc. Không dùng mức thay đổi pixel
lớn hoặc chỉ số Laplacian cao làm bằng chứng duy nhất cho chất lượng.

Nhánh mặc định chỉ tăng chi tiết quan sát được từ ảnh nguồn. Không sinh mắt,
răng, lỗ chân lông; không thay đổi tọa độ pixel bằng làm đẹp, biến dạng, optical
flow hoặc warp lại ảnh màu. Resize để xuất ảnh vẫn được phép, dùng cùng baseline
Lanczos khi so sánh. Tính bất biến hình học trong mã không bảo đảm tuyệt đối cảm
nhận hình dạng: sharpening vẫn có thể tạo viền, lệch tâm cạnh hoặc bóng đôi.

CPU là đích triển khai. Giữ lazy loading, khóa request, snapshot tham số chung
cho ảnh/video/batch và detector mobile. Không training, đổi checkpoint, bật
CodeFormer hoặc neural face upscale mặc định trong kế hoạch này.

## 2. Điểm xuất phát đã kiểm chứng

Baseline thuật toán là commit 922a7cf, hiển thị v3.0.2 từ commit nhãn 5d903e4.
Pure strength 0.50, Natural 0.35; một lượt tăng chi tiết luminance ở độ phân giải
nguồn, ước lượng nhiễu toàn ảnh, chặn cực trị trước/sau resize.

Kết quả 12 ảnh / 60 ca trong SHARPENING_UPDATE.md: tăng nét có ích ở nhóm mờ,
nhưng LPIPS noise/JPEG tăng khoảng 2.09%; ảnh sạch cũng bị xử lý quá mức. CPU
p95 một số kích thước và xử lý 1x tăng. Vì vậy không tiếp tục chỉ tăng strength.

Hai vấn đề ưu tiên: một mức nhiễu toàn ảnh chưa phản ánh từng vùng; chưa phân
biệt cạnh có tín hiệu thật với nhiễu, texture da và cạnh vốn đã sắc.

## 3. Kiến trúc đề xuất

Luồng chính:

`ảnh nguồn → luminance → bản đồ nhiễu/cạnh/chi tiết → gain theo vùng → delta
luminance có giới hạn → resize ảnh và delta → chặn viền/màu → xuất ảnh`

Ảnh màu gốc luôn làm nền. Thực hiện xử lý bằng OpenCV/NumPy float32; không dựa
vào cv2.ximgproc vì môi trường hiện tại không bảo đảm có opencv-contrib.

Các bản đồ dự kiến ở cùng tọa độ nguồn, giá trị hữu hạn [0, 1]:

| Bản đồ | Vai trò |
| --- | --- |
| Noise confidence | Giảm lực ở vùng nhiều nhiễu; tránh coi mọi texture là noise |
| Edge confidence | Ưu tiên cạnh liên tục có tín hiệu, hạn chế các điểm rời rạc |
| Detail support | Chỉ tăng tần số có trong nguồn; loại residual yếu không đáng tin |
| Flat/already-sharp protection | Hạn chế xử lý nền phẳng và cạnh vốn đã sắc |
| Semantic protection, tùy chọn | Bảo vệ da, vùng mắt/miệng, kính và biên khuôn mặt |

Gain dự kiến là strength người dùng nhân các hệ số tin cậy/bảo vệ; không cộng
nhiều lần sharpening độc lập. Feather bản đồ gain để tránh đường ranh xử lý.
Chi tiết có thể mạnh hơn ở tóc/vải, nhưng mắt/miệng phải bảo thủ và chỉ dùng tín
hiệu nguồn, không mặc định coi chúng là nơi cần tăng lực mạnh nhất.

Nhánh parsing chân dung độc lập với CodeFormer. Pure hiện trả về trước khi tạo
FaceRestoreHelper; phải tách khởi tạo detector/parser nếu cần, không bật lại
face reconstruction để lấy mask. Chỉ warp mask vào ảnh nguồn, không warp ảnh
màu. Vùng che khuất/độ tin cậy thấp được bảo vệ. Nhiều mask chồng nhau lấy mức
bảo vệ lớn hơn, không cộng dồn mức tăng nét.

## 4. Các pha thực hiện và điều kiện hoàn thành

### P0 — Đóng băng baseline, dữ liệu và tiêu chí

- Lưu config, code/dependency/model-metric hashes, seed, ảnh và số đo baseline.
- Giữ 6 ảnh phát triển và 12 ảnh kiểm tra trước làm bộ hồi quy đã quan sát.
  Không gọi 12 ảnh đó là tập kiểm tra mới chưa từng xem.
- Kiểm kê validation còn lại; dự kiến chọn 24–30 ảnh phát triển và 24–30 ảnh
  kiểm tra nội bộ chưa dùng. Chỉ chốt số lượng sau kiểm kê dữ liệu hợp lệ.
- Giữ các biến thể cùng ảnh nguồn trong cùng nhóm. Kiểm tra trùng ảnh, dùng
  thông tin nhóm/quarantine đã có; không tự tuyên bố đã tách danh tính hoàn toàn.
- Bao phủ chân dung nghiêng, kính, tóc/da, nhiều mặt, mặt nhỏ, chữ/sản phẩm và
  cảnh vật nếu dữ liệu hiện có đáp ứng; bổ sung fixture hợp lệ khi thiếu.
- Suy giảm: mờ nhẹ/vừa, motion nhiều hướng, anisotropic blur, nhiễu/JPEG, ảnh
  sạch. Ảnh mờ nặng là nhóm kiểm tra giới hạn, không hứa phục hồi đầy đủ.
- Giữ benchmark 391 ảnh cho quyết định cuối; không dùng để điều chỉnh tham số.

Đầu ra: manifest đã khóa, baseline theo nhóm, config các thử nghiệm và tiêu chí
đã chốt trước khi mở kết quả kiểm tra. Thiếu nhóm nào phải ghi rõ.

### P1 — Nhiễu và độ tin cậy chi tiết theo vùng

- Thay ước lượng một sigma toàn ảnh bằng ước lượng robust trên tile nhỏ, ưu tiên
  vùng gradient thấp; nội suy/feather để tránh đường nối giữa tile.
- Dùng nhiều tín hiệu: residual tần số cao, độ liên tục cạnh và tương phản cục
  bộ. Không coi variance cao là noise hoặc ảnh mờ chỉ từ một ngưỡng Laplacian.
- Lưu riêng các map chẩn đoán; kiểm tra tóc, vải, chữ, da và JPEG block.
- Thử trên development: baseline; chỉ noise map; noise + detail confidence.
  Không thay strength mặc định cùng lúc để tránh mất khả năng xác định nguyên nhân.

Điều kiện qua: giảm khuếch đại noise/JPEG so với baseline; không xóa tóc/chữ,
không có biên tile, không làm mọi vùng thành gain gần 0 khiến ảnh lại như cũ.

### P2 — Làm nét theo cấu trúc với một lượt cộng chi tiết

- So tối đa hai cách tách detail: các dải Gaussian hiện có và guided filter
  self-guided bằng boxFilter; đo chứ không mặc định guided filter tốt hơn.
- Điều khiển fine/medium detail bằng các map P1. Giữ nền luminance tần số thấp,
  dùng noise coring, soft cap và giới hạn độ dốc thay vì clipping mạnh tạo mép.
- Giữ một delta chung cho ba kênh, chặn local extrema sau resize như hiện tại.
- Thử riêng strength 0.35/0.50/0.65 trên development; chọn một cấu hình trước
  khi đánh giá tập kiểm tra. Không mở rộng lưới tham số liên tục theo kết quả.

Điều kiện qua: rõ hơn baseline trên phần lớn ca mờ nhẹ/vừa khi xem 1:1; cạnh
không lệch/nhân đôi; nhóm ảnh sạch không bị sharpen thừa.

### P3 — Bảo vệ vùng chân dung bằng parsing, có điều kiện

- Chỉ triển khai sau khi P1/P2 có baseline ổn định. Đo detector + parser riêng.
- Tái sử dụng parser hiện có, chạy một lần trên từng face crop và dùng lại mask
  trong cùng request. Không cache mask giữa các ảnh khác nhau.
- Mặc định bảo vệ da; giữ mắt, miệng, răng và kính ở mức bảo thủ. Tăng chi tiết
  tóc/áo khi có tín hiệu thật và không bị noise map loại.
- Hạ confidence với crop nhỏ, mặt nghiêng, vùng che khuất hoặc mask bất thường.
- Không phát hiện mặt: dùng P2. Parser thiếu/lỗi: ghi rõ chẩn đoán và dùng nhánh
  cấu trúc đã xác định trước; không báo giả rằng đã bảo vệ theo vùng khuôn mặt.

Điều kiện qua: tăng lợi ích hoặc giảm lỗi chân dung rõ ràng so với P2, và qua
ngân sách CPU. Nếu không qua, không đưa parsing vào đường mặc định; giữ nhánh
thử nghiệm hoặc bỏ phần này. Không tăng ngân sách âm thầm để giữ candidate.

### P4 — Khử mờ nhẹ có điều kiện, tách khỏi sharpening

- Chỉ bắt đầu nếu P2/P3 còn thiếu độ rõ trên nhóm mờ nhẹ/vừa.
- Thử kernel hữu hạn đã định trước trên development, kiểm tra độ tin cậy bằng
  residual/tái làm mờ, noise gain và ringing; residual thấp không tự chứng minh
  kernel đúng. Ảnh không có tham chiếu phải có đánh giá trực quan riêng.
- Chỉ dùng ảnh nguồn, giới hạn iteration và strength. Không đủ confidence thì
  giữ nhánh sharpening; không áp Gaussian deconvolution cho mọi ảnh.
- Tách toàn bộ phép thử khỏi thay đổi geometry/paste và khỏi AI detail fusion.

Điều kiện qua: vượt P2/P3 trên nhóm mục tiêu, không tạo bóng đôi/viền sáng, không
gây mất chi tiết ở nhóm sạch/nhiễu. Không đạt thì dừng P4 và phát hành P2/P3 nếu
chúng độc lập đạt gate; không để P4 trở thành lý do kéo dài vô hạn.

### P5 — Kiểm tra độc lập, hồi quy và CPU

- Chỉ mở tập kiểm tra sau khi khóa candidate/config. Nếu thất bại, lưu nguyên
  kết quả; đưa ca lỗi vào bộ hồi quy và cần tập kiểm tra mới cho lần chọn sau.
- So ảnh nguồn/Lanczos, baseline v3.0.2 và candidate ở cùng kích thước. Crop
  khuôn mặt/texture và toàn ảnh đều phải được xem, tránh nền che mất lỗi mặt.
- Tính PSNR/SSIM/LPIPS, ArcFace coverage và worst-case; ArcFace chỉ hỗ trợ đánh
  giá, không chứng minh tuyệt đối giữ nét mặt. Với nhiều mặt, kiểm từng mặt.
- Bootstrap theo ảnh nguồn, không tính các biến thể là các mẫu độc lập.
- A/B che tên phương án trên ít nhất 50 ca; ghi rõ rõ hơn/giữ cấu trúc/halo/noise.
  Không tự điền lựa chọn thay người đánh giá; ca hòa tính là không thắng.
- Hồi quy: cạnh mềm/sắc, chữ nhỏ, gradient, màu bão hòa, nền phẳng/noise, kích
  thước lẻ/tiny, nhiều mặt, lỗi parser, 1x/2x/4x/8x và giới hạn ảnh lớn.
- Ảnh/video/batch dùng cùng snapshot tham số. Video phải kiểm nhấp nháy gain/mask;
  nếu cần làm mượt theo thời gian, giữ state riêng từng video, reset khi đổi cảnh.
- Đo CPU trên Windows và Linux Space: cold/warm riêng, một/nhiều mặt, median/p95
  từ ít nhất 30 lượt sau warmup, peak RSS. Không chạy benchmark cùng tác vụ nặng.

Đầu ra: báo cáo pass/fail từng nhóm, gallery A/B, log lỗi, thông số đã khóa,
latency/RAM và quyết định giữ/loại candidate.

### P6 — Tích hợp UI và phát hành

- Giữ điều khiển người dùng đơn giản: mức làm nét và tùy chọn bảo vệ chân dung
  nếu P3 qua gate. Không đưa các hệ số nội bộ vào luồng mặc định.
- Strength 0 phải là tắt sharpening. Preset đã lưu vẫn đọc được; tham số mới có
  giá trị mặc định tương thích. Không tăng strength của preset cũ âm thầm.
- Lưu phiên bản thuật toán/config vào kết quả; hiển thị app version và mã build
  từ một nguồn duy nhất để tránh lặp lại nhãn version cũ.
- Chỉ chuyển default sau khi P5 đạt. Giữ đường baseline để rollback bằng config.
- Chạy integration + AppTest + CI; push GitHub, xác minh HF nhận đúng commit,
  rồi xác minh runtime SHA RUNNING trùng commit. Không gọi build dở là đã chạy.

## 5. Tiêu chí nghiệm thu đề xuất, khóa ở P0

Các số dưới đây là mục tiêu kỹ thuật của kế hoạch, chưa phải kết quả đạt được.

| Hạng mục | Gate dự kiến so với v3.0.2 |
| --- | --- |
| Độ rõ cảm nhận | >=70% thắng trên >=50 ca A/B; ca hòa không tính thắng |
| LPIPS nhóm mờ nhẹ/vừa | Median giảm tương đối >=5%; báo riêng motion |
| PSNR/SSIM nhóm mờ | Median không giảm quá 0.2 dB / 0.005 |
| Ảnh sạch | Median LPIPS tăng tuyệt đối <=0.001, PSNR giảm <=0.2 dB |
| Nhiễu/JPEG | Median LPIPS/SSIM không kém baseline; flat-noise std tăng <=3% |
| Giữ nét mặt | Median ArcFace delta >=-0.001, worst >=-0.005; coverage không giảm |
| Cấu trúc | Không có ca méo/đổi bộ phận/nhân đôi nét được xác nhận trong review |
| Cạnh chuẩn tổng hợp | Tâm cạnh lệch <=0.25 pixel ở tọa độ nguồn; overshoot mới <=1/255 |
| Màu sắc nhánh mặc định | Delta B/G/R giống nhau so với cùng Lanczos baseline |
| CPU mặc định | Median và p95 tăng <=10% ở từng nhóm kích thước đã khóa |
| Bộ nhớ | Peak RSS tăng <=15%, đồng thời dưới 75% RAM khả dụng đã đo trên Space |

Không dùng trung bình toàn bộ để che lỗi một nhóm. Gate thị giác và kỹ thuật
cùng bắt buộc; không đổi ngưỡng sau khi thấy candidate thất bại. Mức RAM dành
cho request đồng thời phải được tính trong giới hạn Space, không chỉ một ảnh.

## 6. File/API và cách chia thay đổi

- `wink_enhancer.py`: tách helper phân tích/detail/gain; tiếp tục một hàm public
  apply_adaptive_sharpen. Các tham số hiện có giữ nguyên; tham số mới là keyword
  tùy chọn. Validate shape/dtype/range của map thay vì broadcast ngầm.
- `pipeline.py`: điều phối lựa chọn phiên bản; lazy detector/parser riêng nếu
  P3 cần; không thêm đường vòng tải CodeFormer, không phá khóa/cache/snapshot.
- `app.py`: preset, control đơn giản, metadata phiên bản kết quả, giữ cấu hình
  photo/video/batch nhất quán. Không gọi compute từ background thread vào UI.
- `tools/validate_sharpening.py`: nâng thành harness nhận manifest/config, ghi
  coverage/worst-case/confidence interval và dữ liệu A/B; không cố định mãi 12 ảnh.
- `tools/test_reliability.py`, `tools/test_pipeline.py`: chỉ thêm test hành vi rủi
  ro nêu trên; giữ kiểm tra model thật, không thay bằng toàn bộ mock.

Chia commit theo pha: bộ đo → map → filter → portrait protection nếu đạt → kiểm
chứng → default/UI. Mỗi commit triển khai phải có số đo/test tương ứng. Không
gộp Kaggle pilot, training hoặc INT8 vào các commit này.

## 7. Thứ tự ưu tiên và điểm dừng

P0 → P1 → P2 là gói thực hiện đầu tiên. P3/P4 chỉ tiếp tục khi số đo chứng minh
cần thiết và có lợi. P5/P6 bắt buộc trước đổi mặc định. Không đặt lịch hoàn thành
cứng trước khi có profiling và thời gian review A/B; nỗ lực tập trung vào gói đầu
thay vì mở đồng thời nhiều model/thuật toán.

Thành công là ảnh rõ hơn nhưng vẫn giữ cấu trúc, qua gate từng nhóm và chạy được
trên CPU mục tiêu. Nếu chỉ số nét tăng mà ảnh trông giả, noise nổi hơn hoặc khuôn
mặt thay đổi, candidate bị loại. Giới hạn phục hồi chi tiết đã mất vẫn phải ghi
rõ; không đổi tên preset để biến kết quả chưa đạt thành lời hứa chất lượng.
