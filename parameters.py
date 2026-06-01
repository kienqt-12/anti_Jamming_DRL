# --- CÁC THÔNG SỐ VỀ MÔI TRƯỜNG & HỆ THỐNG TRUYỀN THÔNG ---
nu = 0.1                # Xác suất chuyển đổi trạng thái của kẻ gây nhiễu (mô hình Markov)
arrival_rate = 3        # Tốc độ dữ liệu đến trung bình (dùng cho phân phối Poisson)
nu_p = [0.6, 0.2, 0.2]  # Phân bố xác suất cho các biến động ngẫu nhiên của môi trường
d_t = 4                 # Số lượng gói dữ liệu tối đa truyền đi ở điều kiện bình thường (hành động 1)
e_t = 1                 # Năng lượng tiêu hao để truyền 1 gói dữ liệu

# Các mảng thông số mô phỏng sự ngẫu nhiên của kênh truyền khi chống nhiễu
d_bj_arr = [1, 2, 3]    # Các mức dữ liệu truyền được khi dùng hành động chống nhiễu (hành động 3)
e_hj_arr = [1, 2, 3]    # Điểm thưởng thu thập được khi chọn hành động sạc/thu năng lượng (hành động 2)
dt_ra_arr = [2, 1, 0]   # Các mức giới hạn truyền dữ liệu ngẫu nhiên cho hành động 4 và 5

# Giới hạn vật lý của hệ thống
d_queue_size = 10       # Dung lượng lưu trữ tối đa của hàng đợi dữ liệu (Data Queue)
e_queue_size = 10       # Dung lượng lưu trữ tối đa của pin/năng lượng (Energy Queue)
b_dagger = 3            # Ngưỡng dữ liệu (threshold) giới hạn khi truyền trong điều kiện bị nhiễu

# --- CÁC THÔNG SỐ VỀ THUẬT TOÁN HỌC TĂNG CƯỜNG (DRL) ---
num_actions = 7         # Tổng số hành động Agent có thể thực hiện (từ 0 đến 6)

# Thông số cho thuật toán Q-Learning cơ bản
learning_rate_Q = 0.1   # Tốc độ học (alpha): xác định mức độ ghi đè thông tin mới lên cũ
gamma_Q = 0.9           # Hệ số giảm giá (discount factor): mức độ coi trọng phần thưởng trong tương lai

# Thông số cho thuật toán Deep Q-Learning (DQN) - dành cho việc mở rộng sau này
learning_rate_deepQ = 0.0001 # Tốc độ học của mạng Neural trong DQN
gamma_deepQ = 0.99      # Hệ số giảm giá cho DQN
num_features = 3        # Số lượng đặc trưng đầu vào trạng thái (ở đây là 3: jammer, data, energy)
memory_size = 10000     # Kích thước bộ nhớ trải nghiệm (Replay Buffer) lưu lịch sử để huấn luyện DQN
batch_size = 32         # Kích thước 1 mẻ dữ liệu (batch) lấy ra từ Replay Buffer để học
update_target_network = 5000 # Số bước để copy trọng số cập nhật mạng mục tiêu (Target Network)

# Không gian trạng thái
# Tổng số trạng thái = (số trạng thái jammer) * (số mức dữ liệu) * (số mức năng lượng)
num_states = 2 * (d_queue_size + 1) * (e_queue_size + 1)

# --- CÁC THÔNG SỐ ĐIỀU KHIỂN QUÁ TRÌNH HUẤN LUYỆN ---
step = 1000             # Số bước lặp để in kết quả (reward trung bình) ra màn hình
T = 1000000             # Tổng số vòng lặp (iterations) huấn luyện của thuật toán