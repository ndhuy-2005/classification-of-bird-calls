# 🐦 Bird Species Classification

## 📌 Giới thiệu

Đồ án xây dựng một hệ thống **phân loại loài chim từ hình ảnh** sử dụng các kỹ thuật **Deep Learning và Computer Vision**. Mục tiêu của dự án là xây dựng mô hình có khả năng nhận diện và phân loại hình ảnh chim vào đúng nhóm loài tương ứng.

Dự án bao gồm các bước chính từ **tiền xử lý dữ liệu, khám phá dữ liệu, xây dựng mô hình, huấn luyện, đánh giá và trực quan hóa kết quả**.

---

## 🎯 Mục tiêu

* Xây dựng mô hình phân loại hình ảnh các loài chim.
* Thực hiện tiền xử lý và chuẩn hóa dữ liệu hình ảnh.
* Áp dụng Deep Learning cho bài toán Image Classification.
* Đánh giá hiệu năng mô hình bằng các chỉ số phù hợp.
* Trực quan hóa quá trình huấn luyện và kết quả dự đoán.
* Thử nghiệm khả năng dự đoán trên hình ảnh mới.

---

## 🧠 Phương pháp

Quy trình thực hiện dự án:

```text
Bird Image Dataset
        ↓
Data Preprocessing
        ↓
Image Resizing & Normalization
        ↓
Data Augmentation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction
```

Mô hình được huấn luyện trên tập dữ liệu hình ảnh chim với nhiều lớp tương ứng với các loài khác nhau.

Các bước tiền xử lý bao gồm:

* Resize hình ảnh về kích thước đầu vào phù hợp.
* Normalize pixel values.
* Data Augmentation nhằm tăng tính đa dạng của dữ liệu.
* Chia dữ liệu thành các tập **Training / Validation / Testing**.

---

## 🛠️ Công nghệ sử dụng

| Công nghệ          | Mục đích                                   |
| ------------------ | ------------------------------------------ |
| Python             | Ngôn ngữ lập trình chính                   |
| TensorFlow / Keras | Xây dựng và huấn luyện Deep Learning model |
| NumPy              | Xử lý dữ liệu số                           |
| Pandas             | Phân tích và quản lý dữ liệu               |
| Matplotlib         | Trực quan hóa dữ liệu và kết quả           |
| Seaborn            | Visualization                              |
| OpenCV             | Xử lý hình ảnh                             |

---

## 📊 Đánh giá mô hình

Mô hình được đánh giá dựa trên các chỉ số:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**

Ngoài ra, quá trình huấn luyện được trực quan hóa thông qua:

* Training / Validation Accuracy
* Training / Validation Loss
* Confusion Matrix
* Sample Predictions

---

## 🔍 Kết quả

Mô hình có khả năng nhận diện và phân loại các hình ảnh chim thuộc các lớp khác nhau.

Một số kết quả được sử dụng để đánh giá mô hình:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

Các biểu đồ và kết quả dự đoán được lưu trong thư mục:

```text
results/
```

> **Note:** Các thông số Accuracy, Precision, Recall và F1-score có thể được cập nhật trực tiếp dựa trên kết quả thực nghiệm cuối cùng của mô hình.

---

## 📂 Cấu trúc thư mục

```text
Bird-Species-Classification/
│
├── dataset/
│
├── notebooks/
│   └── bird_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   └── training_loss.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Cài đặt

Clone repository:

```bash
git clone https://github.com/USERNAME/Bird-Species-Classification.git
cd Bird-Species-Classification
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

---

## ▶️ Huấn luyện mô hình

Chạy chương trình training:

```bash
python src/train.py
```

Sau khi huấn luyện, model được lưu vào thư mục:

```text
models/
```

---

## 🔮 Dự đoán

Để thực hiện dự đoán trên một hình ảnh mới:

```bash
python src/predict.py --image path/to/image.jpg
```

Hệ thống sẽ trả về **loài chim được dự đoán** cùng với xác suất dự đoán của mô hình.

---

## 📚 Kiến thức áp dụng

Dự án giúp áp dụng các kiến thức:

* Machine Learning
* Deep Learning
* Computer Vision
* Image Classification
* Data Preprocessing
* Data Augmentation
* Model Evaluation
* Data Visualization

---

## 👨‍💻 Tác giả

**Nguyễn Đình Huy**

Information Technology Student
Ho Chi Minh City University of Technology and Education (HCMUTE)

---

## 📄 License

This project is developed for **academic and educational purposes**.
