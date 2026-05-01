#include <iostream>
#include <vector>

using namespace std;

// Xây dựng lớp mô hình SVM Phân loại đơn giản
class SimpleLinearSVM {
private:
    double w1, w2;        // Trọng số cho 2 đặc trưng: Diện tích (w1) và Giá nhà (w2)
    double b;             // Độ lệch (Bias)
    double learning_rate; // Tốc độ học
    int epochs;           // Số vòng lặp huấn luyện
    double lambda;        // Tham số điều chuẩn (Regularization) - Giúp margin rộng hơn

public:
    SimpleLinearSVM(double lr = 0.001, int ep = 1000, double lam = 0.01) {
        w1 = 0.0;
        w2 = 0.0;
        b = 0.0;
        learning_rate = lr;
        epochs = ep;
        lambda = lam;
    }

    // Hàm huấn luyện
    void fit(const vector<double>& X_dien_tich, const vector<double>& X_gia, const vector<int>& y) {
        int n = y.size();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                
                // 1. Tính toán vị trí của điểm dữ liệu so với ranh giới
                // Nếu condition >= 1: Điểm dữ liệu nằm đúng phe và an toàn ngoài Margin
                // Nếu condition < 1: Điểm dữ liệu nằm sai phe hoặc kẹt trong Margin (Support Vector)
                double condition = y[i] * (w1 * X_dien_tich[i] + w2 * X_gia[i] + b);

                // 2. Cập nhật trọng số (Quy tắc Hinge Loss của SVM)
                if (condition >= 1) {
                    // Dự đoán TỐT: Chỉ tinh chỉnh nhẹ để làm rộng Margin (không quan tâm dữ liệu)
                    w1 = w1 - learning_rate * (2 * lambda * w1);
                    w2 = w2 - learning_rate * (2 * lambda * w2);
                } else {
                    // Dự đoán SAI hoặc QUÁ GẦN RANH GIỚI: Bị phạt nặng, phải xoay đường ranh giới
                    w1 = w1 - learning_rate * (2 * lambda * w1 - X_dien_tich[i] * y[i]);
                    w2 = w2 - learning_rate * (2 * lambda * w2 - X_gia[i] * y[i]);
                    b = b - learning_rate * (-y[i]);
                }
            }
        }
    }

    // Hàm dự đoán khu vực
    int predict(double dien_tich, double gia) {
        // Tính toán xem điểm mới rơi vào bên Âm hay bên Dương của đường ranh giới
        double result = w1 * dien_tich + w2 * gia + b;
        if (result > 0) {
            return 1;  // Thuộc Nhóm 1 (TP.HCM)
        } else {
            return -1; // Thuộc Nhóm -1 (Hà Nội)
        }
    }
};

int main() {
    // Dữ liệu giả lập rút gọn (Trong thực tế bạn load 2000 mẫu từ file)
    // Đặc trưng 1: Diện tích (đã chia 100 để chuẩn hóa)
    vector<double> X_dien_tich = {0.5, 0.6, 0.45, 0.9, 1.2, 1.5}; // 50m2, 60m2, 45m2...
    // Đặc trưng 2: Giá nhà (Tỷ VNĐ)
    vector<double> X_gia =       {6.5, 8.0, 5.0,  3.5, 4.0, 4.5}; 
    // Nhãn (Label): 1 là TP.HCM (Giá thường cao hơn cùng diện tích), -1 là Hà Nội
    vector<int> y =              {1,   1,   1,   -1,  -1,  -1}; 

    SimpleLinearSVM model(0.01, 1000, 0.01);

    cout << "Dang huan luyen SVM Phan Loai...\n";
    model.fit(X_dien_tich, X_gia, y);

    // Thử dự đoán một căn nhà mới
    double dt_moi = 0.55; // 55 m2
    double gia_moi = 7.0; // 7.0 Tỷ
    
    int ket_qua = model.predict(dt_moi, gia_moi);

    cout << "------------------------------------------\n";
    cout << "Nha 55m2, gia 7 Ty VND duoc mo hinh du doan nam o:\n";
    if (ket_qua == 1) {
        cout << "=> TP. HO CHI MINH\n";
    } else {
        cout << "=> HA NOI\n";
    }
    cout << "------------------------------------------\n";

    return 0;
}