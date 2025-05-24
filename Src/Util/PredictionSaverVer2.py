import pandas as pd
import numpy as np
import os # Import thư viện os để làm việc với đường dẫn file

class PredictionLogger:
    """
    A class to log and save model predictions along with the test data.

    It saves misclassified samples for each model and a combined CSV
    of all test data and predictions from all logged models.
    """

    def __init__(self, X_test, Y_test):
        """
        Initializes the PredictionLogger with the test feature data and true labels.

        Args:
            X_test (pd.DataFrame): The feature data for the test set.
            Y_test (pd.Series or np.ndarray): The true labels for the test set.
                                                Should have the same index or length as X_test.
        """
        self.X_test = X_test
        # Đảm bảo Y_test là Series và reset index để căn chỉnh với X_test
        if isinstance(Y_test, pd.Series):
            self.Y_test = Y_test.reset_index(drop=True)
        elif isinstance(Y_test, np.ndarray):
            self.Y_test = pd.Series(Y_test, name='True_Y') # Gán tên mặc định
        else:
            raise TypeError("Y_test must be a pandas Series or numpy array.")

        if len(self.X_test) != len(self.Y_test):
             raise ValueError("X_test and Y_test must have the same number of samples.")

        # Dictionary để lưu trữ Y_predict cho từng mô hình
        self._predictions = {}

        print(f"PredictionSaver initialized with {len(self.X_test)} test samples.")

    def add_model_predictions(self, model_name, Y_predict, save_misclassified=True, misclassified_filename=None):
        """
        Adds predictions from a single model and optionally saves misclassified samples.

        Args:
            model_name (str): The name of the model (used for column names and default filenames).
            Y_predict (pd.Series or np.ndarray): The predicted labels for the X_test data.
                                                  Must have the same length as X_test.
            save_misclassified (bool): If True, saves misclassified samples to a separate CSV.
                                       Defaults to True.
            misclassified_filename (str, optional): Custom filename for the misclassified CSV.
                                                   If None, a default name based on model_name is used.
        """
        # Đảm bảo Y_predict là Series và reset index để căn chỉnh
        if isinstance(Y_predict, pd.Series):
            Y_predict_series = Y_predict.reset_index(drop=True)
        elif isinstance(Y_predict, np.ndarray):
            Y_predict_series = pd.Series(Y_predict, name=f'Predicted_Y_{model_name}')
        else:
            raise TypeError("Y_predict must be a pandas Series or numpy array.")

        if len(Y_predict_series) != len(self.X_test):
             raise ValueError(f"Length of Y_predict for '{model_name}' ({len(Y_predict_series)}) must match the length of X_test ({len(self.X_test)}).")

        # Lưu trữ dự đoán vào dictionary nội bộ
        if model_name in self._predictions:
             print(f"Warning: Predictions for model '{model_name}' already exist and will be overwritten.")
        self._predictions[model_name] = Y_predict_series

        print(f"Predictions for model '{model_name}' added.")

        # Lưu các mẫu dự đoán sai nếu được yêu cầu
        if save_misclassified:
            # Tìm các vị trí mà dự đoán không khớp với nhãn thật
            misclassified_indices = self.Y_test != Y_predict_series

            if not misclassified_indices.any():
                print(f"No misclassified samples found for model '{model_name}'. Skipping misclassified CSV save.")
                return # Không có mẫu sai, không cần lưu file

            # Lọc dữ liệu test và nhãn thật/dự đoán cho các mẫu sai
            misclassified_X = self.X_test[misclassified_indices].copy()
            misclassified_Y_true = self.Y_test[misclassified_indices]
            misclassified_Y_pred = Y_predict_series[misclassified_indices]

            # Tạo DataFrame chứa dữ liệu của các mẫu dự đoán sai
            misclassified_df = misclassified_X.copy()
            # Thêm cột nhãn thật và nhãn dự đoán
            misclassified_df['True_Y'] = misclassified_Y_true
            misclassified_df[f'Predicted_Y_{model_name}'] = misclassified_Y_pred

            # Xác định tên file
            if misclassified_filename is None:
                # Tạo tên file mặc định từ tên mô hình, thay thế khoảng trắng bằng gạch dưới và chuyển sang chữ thường
                filename_to_save = f'misclassified_{model_name.replace(" ", "_").lower()}.csv'
            else:
                filename_to_save = misclassified_filename

            # Lưu DataFrame vào file CSV
            print(f"Saving misclassified samples for '{model_name}' to {filename_to_save}...")
            # Lưu index để có thể liên kết lại với dữ liệu gốc nếu cần
            misclassified_df.to_csv(filename_to_save, index=True)
            print("Saved.")

    def save_all_predictions(self, filename='all_predictions.csv'):
        """
        Saves the complete X_test, Y_test, and predictions from all added models
        into a single CSV file.

        Args:
            filename (str, optional): Custom filename for the combined CSV.
                                      Defaults to 'all_predictions.csv'.
        """
        if not self._predictions:
            print("No model predictions have been added yet. Cannot save all predictions.")
            return

        print(f"Preparing data for saving all predictions...")

        # Bắt đầu DataFrame với dữ liệu X_test
        all_data_df = self.X_test.copy()

        # Thêm cột nhãn thật (Y_test)
        # self.Y_test đã được đảm bảo là Series có tên 'True_Y' và reset index
        all_data_df['True_Y'] = self.Y_test # Tên cột sẽ là 'True_Y'

        # Thêm cột dự đoán cho từng mô hình đã lưu
        for model_name, Y_predict_series in self._predictions.items():
             # Tên cột dự đoán sẽ là Predicted_Y_[Tên_Mô_Hình]
            all_data_df[f'Predicted_Y_{model_name}'] = Y_predict_series

        # Lưu DataFrame tổng hợp vào file CSV
        print(f"Saving all test data and predictions to {filename}...")
        # Lưu index để có thể liên kết lại với dữ liệu gốc nếu cần
        all_data_df.to_csv(filename, index=True)
        print("Saved.")

# --- Hướng dẫn sử dụng Class ---

# Sau khi đã có X_test, Y_test và huấn luyện các mô hình (ví dụ từ bước trước)

# Bước 1: Khởi tạo PredictionLogger với X_test và Y_test
# Giả sử X_test và Y_test của bạn đã sẵn sàng
# logger = PredictionLogger(X_test, Y_test)

# Bước 2: Thêm dự đoán của từng mô hình vào logger
# Sử dụng vòng lặp hoặc thêm từng mô hình một
# Ví dụ:
# for model_name, model_pipeline in models.items(): # 'models' là dictionary các pipeline đã train từ bước trước
#     print(f"\nAdding predictions for {model_name} to logger...")
#     try:
#         Y_pred = model_pipeline.predict(X_test) # Lấy dự đoán từ mô hình đã train
#         # Thêm dự đoán vào logger. Mặc định sẽ lưu file misclassified riêng
#         logger.add_model_predictions(model_name, Y_pred)
#     except Exception as e:
#          print(f"Could not get predictions for {model_name}: {e}")


# Bước 3: (Tùy chọn) Lưu dự đoán sai của một mô hình với tên file tùy chỉnh
# Ví dụ:
# logger.add_model_predictions("Support Vector Machine (SVM)", Y_pred_svm, misclassified_filename="svm_wrong_predictions.csv")

# Bước 4: Lưu toàn bộ dữ liệu test và dự đoán của tất cả mô hình vào một file CSV chung
# logger.save_all_predictions(filename="combined_model_predictions.csv") # Đặt tên file tùy chỉnh
# Hoặc để tên mặc định:
# logger.save_all_predictions() # File sẽ tên là all_predictions.csv

# Các file CSV sẽ được tạo trong cùng thư mục với script Python của bạn.
