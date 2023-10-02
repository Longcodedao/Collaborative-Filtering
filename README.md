# Collaborative Filtering


## 1. Giới thiệu

Lọc Cộng tác (Collaborative Filtering) là một phương pháp trong lĩnh vực hệ thống đề xuất (Recommender Systems) trong khoa học máy tính và thống kê. Phương pháp được sử dụng với mục đích đề xuất các mục (hay còn gọi là items) hoặc nội dung trên thông tin và mức độ quan tâm tương đồng của từng *users*. *Mức độ tương đồng* được xác địch bằng các đánh giá của từng *users* đối với các items trong toàn bộ hệ thống. Ví dụ, *anh A* và *chị B* đều yêu thích máy tính Iphone 15 Pro Max và đánh giá 5 sao trên FPT Shop. Ta cũng đã biết *anh A* cũng thích máy tính Macbook M2 Pro, vậy có khả năng cao là *chị B* cũng thích sản phẩm này.

Một trong những câu hỏi quan trọng nhất trong Collaborative Filtering là:
    - Làm sao xác định được sự tương đồng giữa 2 *users* khác nhau?
    - Khi đã xác định được các *users* giống nhau, làm sao có thể dự đoán mức độ quan tâm của một user trên một item.

Có 2 cách tiếp cận cho bài toán Lọc Cộng tác này. Cách thứ nhất là xác định mức độ quan tâm của một user dựa trên những user khác giống nhau, hay còn gọi là Lọc Cộng tác dựa trên Người dùng (User-User Collaborative Filtering). Còn có một cách tiếp cận khác là xác định những items gần giống với những items có sự ưa chuộng cao (Item-Item Collaborative Filtering)



## 2. Lọc cộng tác dựa trên Người Dùng (User-User Collaborative Filtering)

### 2.1 Similarity functions

Việc xác định được độ giống nhau của những *users* là bước quan trọng nhất trong phương pháp User-User Collaborative Filtering. Dữ liệu hiện tại đang có là 1 ma trận *Utility* $Y$, ma trận thể hiện đánh giá (rating) của từng user cho tưng items khác nhau. 

![Hình 1: Ma trận Utility $Y$](https://github.com/Longcodedao/Collaborative-Filtering/blob/main/images/utility.png?raw=true)

Các con số trong ma trận này là thang đo đánh giá từ 1 đến 5 sao, còn các dấu hỏi là những mục mà chưa có đánh giá (có thể là do người dùng chưa biết những sản phẩm này). Đó chính là những giá trị mà hệ thống phải đi tìm để gợi ý. Dựa vào mắt thường, ta có thể nhận biết các đánh giá user $u_0$ gần giống với user $u_1$ hơn các user còn lại. Vậy còn công thức toán học thì sao? Ta có công thức thể hiện mức động giống nhau của hai *users* $u_i$ và $u_j$ được kí hiệu là $sim(u_i, u_j)$. 

Trong trường hợp này, $u_0$ và $u_1$ đều thích items $i_0$ và $i_1$, trái ngược với các user còn lại nên ta có $$\large sim(u_0, u_1) > sim(u_0, u_i),  \forall i > 1$$

Để đo *similarity* giữa hai users, ta thường xây dựng một vector đặt trưng (feature vector) cho từng user (vector gồm từng rating cho mỗi items khác nhau). Ví dụ, vector đặt trưng cho *user* $u_2$ là $[2, ?, 1, 3, 4]$ cho từng items $i_0$, $i_1$, ..., $i_4$. Tuy nhiên, thực tế thì ma trận *Utility* này rất lớn (hệ thống thương mai điện tử  lớn thường có hàng triệu sản phẩm) nhưng mà số lượng rating thì rất ít (mỗi *user* thường rate rất ít). Vì vậy nên dễ dẫn đến vấn đề ma trận thưa (sparsity matrix) khiến cho việc gợi ý trở nên sai lệch và tăng bộ nhớ và khối lượng tính toán. Cách khắc phục là phân rã ma trận (Matrix Factorization) hoặc giảm chiều dữ liệu PCA (Dimensionality Reduction.)

![Hình 2: Ví dụ mô tả User-user Collaborative Filtering. a) Utility Matrix ban đầu. b) Utility Matrix đã được chuẩn hoá. c) User similarity matrix. d) Dự đoán các (normalized) ratings còn thiếu. e) Ví dụ về cách dự đoán normalized rating của $u_1$ cho $i_1$ f) Dự đoán các (denormalized) ratings còn thiếu.](https://github.com/Longcodedao/Collaborative-Filtering/blob/main/images/user_cf.png?raw=true)


**Chuẩn hóa dữ liệu**

Giá trị mỗi rating tương ứng với nếu trừ đi với các giá trị trung bình của *ratings* cho từng user thì ta sẽ được ma trận *utility* được chuẩn hóa (hay còn gọi là $\hat{Y}$) như ở hình 2b). Có một số lý do như sau:

- Trừ đi trung bình cộng của *user* khiến các giá trị cho từng user có thể nhận giá trị dương và âm. Giá trị dương tương ứng với việc *user* đánh giá tích cực về *item* và giá trị âm tương ứng với việc *user* đánh giá tiêu cực về *item* đó. Giá trị 0 tương ứng với việc *user* có góc nhìn *"trung lập"* về *item* đó.

- Ngoài ra, số chiều của ma trận *utility* là rất lớn với hàng triêụ *users* và *items*. Vì vậy, sẽ là một điều bất khả thi nếu lưu toàn bộ ma trận đó (không đủ bộ nhớ). Số lượng *ratings* thường sẽ rất nhỏ so với kích thước của ma trận *utility* nên ta hay lưu dưới dạng *ma trận thưa*, tức là chỉ lưu các giá trị khác 0 và vị trí của chúng. Vì vậy ta sẽ điền các giá trị ? bằng số 0.

**Tương quan Cosine**

*Cosine similarity* hay còn được gọi là *tương tự cosine* là một dạng phép đo thể hiện sự tương đồng giữa hai vector trong không gian đa chiều. Trong trường hợp này, ta sử dụng công thức này để thể hiện sự tương quan của các vector *user*. $$\large Cosine\hspace{3pt}Similarity(\mathbf{u_1}, \mathbf{u_2}) = cos(\mathbf{u_1}, \mathbf{u_2}) = \frac{\mathbf{u_1}^{T}\mathbf{u_2}}{\|\mathbf{u_1}\|\|\mathbf{u_2}\|}$$

Trong đó $\mathbf{u_1}$ và $\mathbf{u_2}$ là vector tương ứng với *user 1* và *user 2* **đã được chuẩn hóa** như trên.

Nếu giải thích dưới góc độ Toán học thì *cosine similarity* đo giá trị góc giữa 2 vector (đã học năm lớp 10). Độ *similarity* của hai vector này là đều nằm trong khoảng $[-1, 1]$ vì $-1 \leq cos(x) \leq 1$. Giá trị bằng 1 có nghĩa là 2 vector này cùng hướng, *similar* với nhau. Giá trị bằng -1 có nghĩa là 2 vector ngược hướng, hoàn toàn trái ngược nhau. Tức là hành vi trái ngược nhau

Từ công thức này, ta có thể dựng một ma trận tương quan *similarity matrix* $\mathbf{S}$ như trên hình 2c)  


### 2.2 Dự đoán Rating

Xác định mức độ quan tâm của một *user* lên một *item* thường dựa trên các *users* gần nhất (*neighbor users*), rất giống với bài toán KNN. Trong Lọc Cộng tác, *missing rate* cũng xác định dựa trên $k$ *neighbor users* và ta chỉ quan tâm đến các *users* đã đánh giá *item* đang xem xét. Giá trị dự đoán rating được xác định bởi trung bình có trọng số của các đánh giá (*ratings*) đã được chuẩn hóa. 

<!-- <center>
![](https://github.com/Longcodedao/Collaborative-Filtering/blob/main/images/CodeCogsEqn.gif?raw=true) 
</center> -->

<p align="center">
  <img src="https://github.com/Longcodedao/Collaborative-Filtering/blob/main/images/CodeCogsEqn.gif?raw=true" alt=""/>
</p>

Trong đó $N(u, i)$ là tập hợp các $k$ *users* có *similarity* cao nhất của $u$ mà đã rate *item* $i$.



