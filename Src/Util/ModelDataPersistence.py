import pandas as pd
import numpy as np
import joblib
import pickle
import os
import warnings # Để cảnh báo nếu format không phù hợp

class ModelDataManager:
    """
    A class to manage saving and loading datasets and trained models.

    Supports various formats (pickle, joblib, csv, parquet) and flexible
    filename handling (automatic based on name or manual).
    """

    # Định nghĩa các định dạng mặc định và extension tương ứng
    _DEFAULT_DATASET_FORMAT = 'pkl'
    _DEFAULT_MODEL_FORMAT = 'joblib'
    _FORMAT_EXTENSIONS = {
        'pkl': '.pkl',
        'joblib': '.joblib',
        'csv': '.csv',
        'parquet': '.parquet'
    }
    _ALLOWED_DATASET_FORMATS = ['pkl', 'csv', 'parquet']
    _ALLOWED_MODEL_FORMATS = ['pkl', 'joblib']

    @staticmethod
    def _get_extension(format):
        """Helper to get file extension for a given format."""
        return ModelDataManager._FORMAT_EXTENSIONS.get(format.lower(), None)

    @staticmethod
    def _determine_format_from_filename(filename):
        """Helper to determine format from filename extension."""
        if not isinstance(filename, str):
            return None
        _, ext = os.path.splitext(filename.lower())
        for fmt, known_ext in ModelDataManager._FORMAT_EXTENSIONS.items():
            if ext == known_ext:
                return fmt
        return None

    @staticmethod
    def _determine_save_path(name, format, filename):
        """Helper to determine the final save path."""
        if filename:
            return filename
        ext = ModelDataManager._get_extension(format)
        if not ext:
            raise ValueError(f"Unsupported format: {format}. Cannot determine file extension.")
        return f"{name}{ext}"

    @staticmethod
    def _determine_load_path(name, filename, format):
        """Helper to determine the final load path and format for loading."""
        if filename:
            # If filename is given, try to determine format from its extension
            # unless format is explicitly specified
            load_format = format if format != 'auto' else ModelDataManager._determine_format_from_filename(filename)
            if not load_format:
                 raise ValueError(f"Could not determine format from filename '{filename}' or specified format '{format}' is invalid.")
            return filename, load_format

        if name is None:
             raise ValueError("Either filename or name must be provided for loading.")

        # If only name is given, try common extensions based on expected format type (dataset/model)
        if format == 'auto':
            # Try common dataset/model extensions
            possible_formats = [ModelDataManager._DEFAULT_MODEL_FORMAT, ModelDataManager._DEFAULT_DATASET_FORMAT] + \
                               [f for f in ModelDataManager._ALLOWED_DATASET_FORMATs + ModelDataManager._ALLOWED_MODEL_FORMATs if f not in [ModelDataManager._DEFAULT_MODEL_FORMAT, ModelDataManager._DEFAULT_DATASET_FORMAT]]

            for fmt in possible_formats:
                 ext = ModelDataManager._get_extension(fmt)
                 if ext:
                     potential_filename = f"{name}{ext}"
                     if os.path.exists(potential_filename):
                         print(f"Auto-detected file: {potential_filename} with format: {fmt}")
                         return potential_filename, fmt
            raise FileNotFoundError(f"Could not find a file for '{name}' with common extensions (.joblib, .pkl, .csv, .parquet). Specify filename or format.")

        else: # Specific format is provided, construct filename
            ext = ModelDataManager._get_extension(format)
            if not ext:
                 raise ValueError(f"Unsupported format: {format}. Cannot determine file extension.")
            potential_filename = f"{name}{ext}"
            if os.path.exists(potential_filename):
                 return potential_filename, format
            else:
                 raise FileNotFoundError(f"File not found: {potential_filename}")


    @staticmethod
    def save_dataset(dataset, name, format=_DEFAULT_DATASET_FORMAT, filename=None, **kwargs):
        """
        Saves a dataset object (DataFrame, Series, or numpy array) to a file.

        Args:
            dataset: The dataset object to save.
            name (str): The logical name of the dataset (used for default filename).
            format (str): The desired storage format ('pkl', 'csv', 'parquet').
                          Defaults to 'pkl'.
            filename (str, optional): Custom filename to save to. If None,
                                      a default name based on 'name' and 'format' is used.
            **kwargs: Additional keyword arguments passed to the underlying save function
                      (e.g., index=False for csv/parquet).
        """
        lower_format = format.lower()
        if lower_format not in ModelDataManager._ALLOWED_DATASET_FORMATS:
            raise ValueError(f"Unsupported dataset format: {format}. Allowed formats are: {ModelDataManager._ALLOWED_DATASET_FORMATS}")

        save_path = ModelDataManager._determine_save_path(name, lower_format, filename)
        print(f"Saving dataset '{name}' to {save_path} in {lower_format} format...")

        try:
            if lower_format == 'pkl':
                if not isinstance(dataset, (pd.DataFrame, pd.Series, np.ndarray)):
                     warnings.warn(f"Saving object of type {type(dataset).__name__} as pkl. Ensure it's serializable.")
                with open(save_path, 'wb') as f:
                    pickle.dump(dataset, f)
            elif lower_format == 'joblib': # Joblib is also viable for numpy, though pkl is more general
                 if not isinstance(dataset, (pd.DataFrame, pd.Series, np.ndarray)):
                      warnings.warn(f"Saving object of type {type(dataset).__name__} as joblib. Ensure it's suitable for joblib.")
                 joblib.dump(dataset, save_path)
            elif lower_format == 'csv':
                if not isinstance(dataset, (pd.DataFrame, pd.Series)):
                    raise TypeError(f"CSV format is only supported for pandas DataFrame or Series, but got {type(dataset).__name__}")
                # Default to index=False for CSV unless specified
                kwargs['index'] = kwargs.get('index', False)
                dataset.to_csv(save_path, **kwargs)
            elif lower_format == 'parquet':
                if not isinstance(dataset, pd.DataFrame):
                     raise TypeError(f"Parquet format is only supported for pandas DataFrame, but got {type(dataset).__name__}")
                # Default to index=False for Parquet unless specified
                kwargs['index'] = kwargs.get('index', False)
                dataset.to_parquet(save_path, **kwargs)
            # elif other formats...

            print(f"Dataset '{name}' saved successfully.")

        except Exception as e:
            print(f"Error saving dataset '{name}' to {save_path}: {e}")
            # Clean up potentially partial file
            if os.path.exists(save_path):
                os.remove(save_path)
            raise # Re-raise the exception


    @staticmethod
    def load_dataset(name=None, filename=None, format='auto', **kwargs):
        """
        Loads a dataset object from a file.

        Args:
            name (str, optional): The logical name of the dataset (used for default filename
                                  if filename is None).
            filename (str, optional): The path to the file to load. If None, a default name
                                      based on 'name' and 'format' is used.
            format (str): The expected format of the file ('auto', 'pkl', 'csv', 'parquet').
                          If 'auto', format is inferred from the file extension. Defaults to 'auto'.
            **kwargs: Additional keyword arguments passed to the underlying load function
                      (e.g., index_col for csv).

        Returns:
            The loaded dataset object.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If format is unsupported or cannot be determined.
            TypeError: If loading fails due to file content or format mismatch.
        """
        load_path, determined_format = ModelDataManager._determine_load_path(name, filename, format)
        print(f"Loading dataset from {load_path} in {determined_format} format...")

        try:
            if determined_format == 'pkl':
                with open(load_path, 'rb') as f:
                    dataset = pickle.load(f)
            elif determined_format == 'joblib':
                 dataset = joblib.load(load_path)
            elif determined_format == 'csv':
                dataset = pd.read_csv(load_path, **kwargs)
            elif determined_format == 'parquet':
                dataset = pd.read_parquet(load_path, **kwargs)
            else:
                 # This case should ideally not be reached due to _determine_load_path checks
                 raise ValueError(f"Loading not implemented for format: {determined_format}")


            print(f"Dataset loaded successfully from {load_path}.")
            return dataset

        except FileNotFoundError:
             raise # Re-raise file not found directly
        except Exception as e:
            print(f"Error loading dataset from {load_path}: {e}")
            raise # Re-raise other exceptions


    @staticmethod
    def save_model(model, name, format=_DEFAULT_MODEL_FORMAT, filename=None):
        """
        Saves a trained model object to a file.

        Args:
            model: The trained model object to save.
            name (str): The logical name of the model (used for default filename).
            format (str): The desired storage format ('joblib', 'pkl'). Defaults to 'joblib'.
            filename (str, optional): Custom filename to save to. If None,
                                      a default name based on 'name' and 'format' is used.
        """
        lower_format = format.lower()
        if lower_format not in ModelDataManager._ALLOWED_MODEL_FORMATS:
            raise ValueError(f"Unsupported model format: {format}. Allowed formats are: {ModelDataManager._ALLOWED_MODEL_FORMATS}")

        save_path = ModelDataManager._determine_save_path(name, lower_format, filename)
        print(f"Saving model '{name}' to {save_path} in {lower_format} format...")

        try:
            if lower_format == 'joblib':
                joblib.dump(model, save_path)
            elif lower_format == 'pkl':
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            # elif other formats...

            print(f"Model '{name}' saved successfully.")

        except Exception as e:
            print(f"Error saving model '{name}' to {save_path}: {e}")
            if os.path.exists(save_path):
                 os.remove(save_path)
            raise # Re-raise


    @staticmethod
    def load_model(name=None, filename=None, format='auto'):
        """
        Loads a trained model object from a file.

        Args:
            name (str, optional): The logical name of the model (used for default filename
                                  if filename is None).
            filename (str, optional): The path to the file to load. If None, a default name
                                      based on 'name' and 'format' is used.
            format (str): The expected format of the file ('auto', 'joblib', 'pkl').
                          If 'auto', format is inferred from the file extension. Defaults to 'auto'.

        Returns:
            The loaded model object.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If format is unsupported or cannot be determined.
            TypeError: If loading fails due to file content or format mismatch.
        """
        load_path, determined_format = ModelDataManager._determine_load_path(name, filename, format)
        print(f"Loading model from {load_path} in {determined_format} format...")

        try:
            if determined_format == 'joblib':
                model = joblib.load(load_path)
            elif determined_format == 'pkl':
                with open(load_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                 # This case should ideally not be reached
                 raise ValueError(f"Loading not implemented for format: {determined_format}")

            print(f"Model loaded successfully from {load_path}.")
            return model

        except FileNotFoundError:
             raise
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
            raise


# --- Hướng dẫn sử dụng Class ---

# Giả sử bạn có các biến sau sau khi tiền xử lý và huấn luyện:
# X_train, Y_train, X_test, Y_test (có thể là DataFrame, Series, hoặc NumPy arrays)
# models (dictionary chứa các mô hình đã huấn luyện)

# --- Lưu trữ ---

# Lưu từng dataset một cách từ từ (mặc định dùng pkl)
# print("\n--- Saving Datasets ---")
# ModelDataManager.save_dataset(X_train, 'X_train')
# ModelDataManager.save_dataset(Y_train, 'Y_train') # Y_train có thể là Series/ndarray, pkl handle tốt
# ModelDataManager.save_dataset(X_test, 'X_test')
# ModelDataManager.save_dataset(Y_test, 'Y_test')

# # Lưu một dataset với định dạng khác (ví dụ CSV) và tên file tùy chỉnh
# # Lưu ý: CSV/Parquet tốt nhất cho DataFrame/Series, không phải NumPy array tùy ý
# # ModelDataManager.save_dataset(X_train, 'X_train', format='csv', filename='my_xtrain_features.csv', index=False) # Lưu ý index=False

# # Lưu các mô hình dùng vòng lặp (mặc định dùng joblib)
# print("\n--- Saving Models ---")
# # Giả sử 'models' là dictionary các mô hình đã train từ bước trước
# # for model_name, model_pipeline in models.items():
# #    ModelDataManager.save_model(model_pipeline, model_name)

# # Lưu một mô hình với định dạng khác (ví dụ pkl) và tên file tùy chỉnh
# # svm_model_trained = models.get("Support Vector Machine (SVM)") # Lấy mô hình đã train
# # if svm_model_trained:
# #     ModelDataManager.save_model(svm_model_trained, "SVM_model", format='pkl', filename='my_svm_pipeline.pkl')


# --- Nạp lại ---

# Nạp các dataset một cách từ từ (mặc định dùng auto format và tên biến)
# print("\n--- Loading Datasets ---")
# try:
#     X_train_loaded = ModelDataManager.load_dataset('X_train')
#     Y_train_loaded = ModelDataManager.load_dataset('Y_train')
#     X_test_loaded = ModelDataManager.load_dataset('X_test')
#     Y_test_loaded = ModelDataManager.load_dataset('Y_test')
#     print("\nDatasets loaded successfully:")
#     print(f"X_train_loaded type: {type(X_train_loaded)}")
#     print(f"Y_train_loaded type: {type(Y_train_loaded)}")
#     print(f"X_test_loaded type: {type(X_test_loaded)}")
#     print(f"Y_test_loaded type: {type(Y_test_loaded)}")

# except FileNotFoundError as e:
#      print(f"Failed to load dataset: {e}")
# except Exception as e:
#      print(f"An error occurred during dataset loading: {e}")


# # Nạp một dataset với tên file tùy chỉnh (tự động nhận format)
# # try:
# #     X_train_loaded_csv = ModelDataManager.load_dataset(filename='my_xtrain_features.csv')
# #     print(f"\nLoaded dataset from custom CSV file. Type: {type(X_train_loaded_csv)}")
# # except Exception as e:
# #      print(f"Failed to load dataset from custom CSV: {e}")


# # Nạp các mô hình dùng vòng lặp (mặc định dùng auto format và tên biến)
# print("\n--- Loading Models ---")
# loaded_models = {}
# # Giả sử bạn có danh sách tên các mô hình cần nạp
# # model_names_to_load = ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
# # for model_name in model_names_to_load:
# #     try:
# #         loaded_model = ModelDataManager.load_model(model_name)
# #         loaded_models[model_name] = loaded_model
# #         print(f"Model '{model_name}' loaded successfully. Type: {type(loaded_model)}")
# #     except FileNotFoundError as e:
# #         print(f"Failed to load model '{model_name}': {e}")
# #     except Exception as e:
# #         print(f"An error occurred during model loading '{model_name}': {e}")

# # Nạp một mô hình với tên file tùy chỉnh và format rõ ràng
# # try:
# #     svm_model_loaded_pkl = ModelDataManager.load_model(filename='my_svm_pipeline.pkl', format='pkl')
# #     print(f"\nLoaded SVM model from custom PKL file. Type: {type(svm_model_loaded_pkl)}")
# # except Exception as e:
# #      print(f"Failed to load SVM model from custom PKL: {e}")