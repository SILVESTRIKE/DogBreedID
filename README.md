# DogBreedID

Đồ án Thực hành Deep Learning lần 1: Nhận diện giống chó bằng Deep Learning và giao diện Streamlit

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Cài đặt thư viện cần thiết](#cài-đặt-thư-viện-cần-thiết)
- [Chạy ứng dụng Streamlit](#chạy-ứng-dụng-streamlit)
- [Cách sử dụng ứng dụng](#cách-sử-dụng-ứng-dụng)
- [Truy cập bản demo trực tuyến](#truy-cập-bản-demo-trực-tuyến)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

DogBreedID là ứng dụng nhận diện giống chó từ ảnh sử dụng Deep Learning, xây dựng giao diện người dùng bằng Streamlit. Người dùng chỉ cần tải ảnh lên, hệ thống sẽ dự đoán giống chó.

## Yêu cầu hệ thống

- Python >= 3.8
- pip (Python package installer)
- Khuyến khích: Tạo môi trường ảo (virtual environment) để quản lý thư viện

## Cài đặt môi trường

**Bước 1:** Clone repo về máy:
```bash
git clone https://github.com/SILVESTRIKE/DogBreedID.git
cd DogBreedID
```

**Bước 2:** (Khuyến khích) Tạo môi trường ảo:
```bash
python -m venv venv
# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

## Cài đặt thư viện cần thiết

Cài đặt các thư viện bằng pip:
```bash
pip install -r requirements.txt
```
**Nếu không có file requirements.txt, bạn có thể cài đặt thủ công:**
```bash
pip install streamlit tensorflow numpy pandas pillow
```
(Tùy vào mã nguồn, có thể thêm `scikit-learn`, `matplotlib`,...)

## Chạy ứng dụng Streamlit

Sau khi cài xong thư viện, chạy lệnh:
```bash
streamlit run app.py
```
Trong đó, `app.py` là file chính chứa code giao diện Streamlit (nếu tên file khác, hãy thay bằng tên file tương ứng).

Sau khi chạy, Streamlit sẽ mở trang web trên trình duyệt, thường tại địa chỉ: http://localhost:8501

## Cách sử dụng ứng dụng

1. Mở giao diện web sau khi chạy lệnh ở trên.
2. Tải lên ảnh một chú chó bằng nút "Browse files" hoặc "Chọn tệp".
3. Ấn nút dự đoán (nếu có), hoặc ứng dụng sẽ tự động hiện kết quả dự đoán giống chó.
4. Xem kết quả dự đoán và các thông tin liên quan.

## Truy cập bản demo trực tuyến

Bạn có thể dùng thử ứng dụng ngay tại đây (không cần cài đặt gì):
👉 [https://dogbreedid.streamlit.app/](https://dogbreedid.streamlit.app/)

## Lưu ý

- Ảnh đầu vào nên rõ nét, có mặt chó nhìn trực diện để tăng độ chính xác.
- Nếu gặp lỗi về thư viện, hãy kiểm tra phiên bản Python và các thư viện đã cài đặt đầy đủ chưa.

## Tài liệu tham khảo

- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)

---

**Mọi thắc mắc/báo lỗi vui lòng tạo issue trên GitHub hoặc liên hệ trực tiếp qua email.**
