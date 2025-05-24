import pandas as pd # Có thể không dùng trực tiếp trong class, nhưng phổ biến trong context
import numpy as np # Có thể không dùng trực tiếp trong class
import joblib # Có thể không dùng trực tiếp trong class
import pickle # Có thể không dùng trực tiếp trong class
import os
import subprocess # Để chạy các lệnh command line
import zipfile # Để giải nén file zip
import tarfile # Để giải nén file tar/tar.gz/tar.bz2
import warnings
from datetime import datetime # Để tạo tên file mặc định có timestamp

class ColabArchiveManager:
    """
    A utility class for compressing and decompressing files/folders in Google Colab.

    Uses command line tools (zip, tar) for flexible compression and Python
    libraries (zipfile, tarfile) for safer decompression. Designed for Colab environment.
    """

    _DEFAULT_COMPRESS_FORMAT = 'zip'
    _ALLOWED_COMPRESS_FORMATS = ['zip', 'tar.gz', 'tar.bz2'] # tar.gz maps to gztar, tar.bz2 maps to bztar
    _DEFAULT_ARCHIVE_NAME = 'archive' # Base name for default archive name
    _DEFAULT_DECOMPRESS_DIR_SUFFIX = '_extracted' # Suffix for default extraction folder

    @staticmethod
    def compress(sources='.', archive_name=None, format=_DEFAULT_COMPRESS_FORMAT, exclude=None):
        """
        Compresses specified files/folders into an archive file.

        Uses command line tools (zip, tar).

        Args:
            sources (str or list): Path(s) to file(s) or folder(s) to compress.
                                   Defaults to '.' (current directory).
                                   Can be a single path or a list of paths.
            archive_name (str, optional): The desired base name for the output archive
                                          (without extension). If None, a default name
                                          based on current time or source name is used.
            format (str): The archive format ('zip', 'tar.gz', 'tar.bz2').
                          Defaults to 'zip'.
            exclude (str or list, optional): Pattern(s) or path(s) to exclude from compression.
                                             e.g., '*.ipynb_checkpoints/*' for zip,
                                             ['path/to/exclude1', 'path/to/exclude2'] for tar.
                                             Support varies by format and command line tool.

        Returns:
            str: The full path to the created archive file.

        Raises:
            ValueError: If format is unsupported or arguments are invalid.
            RuntimeError: If the command line compression fails.
        """
        lower_format = format.lower()
        if lower_format not in ColabArchiveManager._ALLOWED_COMPRESS_FORMATS:
            raise ValueError(f"Unsupported compression format: {format}. Allowed formats are: {ColabArchiveManager._ALLOWED_COMPRESS_FORMATS}")

        # Determine archive base name
        if archive_name is None:
            if isinstance(sources, str) and sources != '.' and not os.path.sep in sources:
                 # If single source is a file/folder name without path separators, use it
                 base_name = sources.split('/')[-1] # Take last part if it contains path (though unlikely)
            else:
                 # Generic name with timestamp for '.' or multiple/pathed sources
                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                 base_name = f"{ColabArchiveManager._DEFAULT_ARCHIVE_NAME}_{timestamp}"
            print(f"Archive name not specified. Using default: {base_name}")
        else:
            base_name = archive_name

        # Determine archive filename with extension
        if lower_format == 'zip':
            archive_filename = f"{base_name}.zip"
            command = ["zip", "-r", archive_filename] # -r for recursive (folders)
        elif lower_format == 'tar.gz':
            archive_filename = f"{base_name}.tar.gz"
            command = ["tar", "-czf", archive_filename] # -c create, -z gzip, -f file
        elif lower_format == 'tar.bz2':
            archive_filename = f"{base_name}.tar.bz2"
            command = ["tar", "-cjf", archive_filename] # -c create, -j bzip2, -f file
        # elif other formats...
        else: # Should not happen due to check above
             raise ValueError(f"Internal error: Format {lower_format} not mapped to command.")

        # Add sources to the command
        if isinstance(sources, str):
            command.append(sources)
        elif isinstance(sources, list):
            if not sources:
                 raise ValueError("Sources list cannot be empty.")
            command.extend(sources)
        else:
            raise TypeError("Sources must be a string path or a list of string paths.")

        # Add exclude patterns/paths (command syntax varies by zip/tar)
        if exclude:
             if lower_format == 'zip':
                  # zip -x expects space-separated patterns
                 if isinstance(exclude, str):
                     command.extend(["-x", exclude])
                 elif isinstance(exclude, list):
                     for pattern in exclude:
                          command.extend(["-x", pattern])
                 else:
                      warnings.warn(f"Exclude for zip expects string or list of strings. Got {type(exclude).__name__}. Skipping exclude.")
             elif lower_format.startswith('tar'):
                  # tar --exclude expects one exclude per argument
                 if isinstance(exclude, str):
                     command.extend(["--exclude", exclude])
                 elif isinstance(exclude, list):
                     for pattern in exclude:
                          command.extend(["--exclude", pattern])
                 else:
                      warnings.warn(f"Exclude for tar expects string or list of strings. Got {type(exclude).__name__}. Skipping exclude.")
             # elif other formats...


        print(f"Running compression command: {' '.join(command)}")

        # Execute the command
        try:
            # Use shell=False for better security and handling of spaces in paths
            # Pass command as a list of arguments
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Compression successful.")
            # print("Stdout:", result.stdout) # Optional: print command output
            # print("Stderr:", result.stderr) # Optional: print command errors/warnings

        except FileNotFoundError:
             raise RuntimeError(f"Compression command not found. Is '{command[0]}' installed? (Usually available in Colab)")
        except subprocess.CalledProcessError as e:
            print(f"Compression command failed with exit code {e.returncode}")
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
            # Clean up potentially partial file
            if os.path.exists(archive_filename):
                 os.remove(archive_filename)
            raise RuntimeError(f"Compression failed: {e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred during compression: {e}")
            if os.path.exists(archive_filename):
                 os.remove(archive_filename)
            raise RuntimeError(f"Compression failed: {e}")


        # Return the full path (usually in the current directory)
        return os.path.abspath(archive_filename)

    @staticmethod
    def decompress(archive_path, dest_dir=None):
        """
        Decompresses an archive file into a destination directory.

        Uses Python's standard libraries (zipfile, tarfile) for safety.

        Args:
            archive_path (str): The path to the archive file (.zip, .tar, .tar.gz, .tar.bz2).
            dest_dir (str, optional): The destination directory for extraction. If None,
                                      a new folder named after the archive (without extension)
                                      is created in the current directory.

        Returns:
            str: The path to the destination directory.

        Raises:
            FileNotFoundError: If the archive file is not found.
            ValueError: If the archive format is unsupported or archive is invalid.
            RuntimeError: If decompression fails.
        """
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive file not found: {archive_path}")

        # Determine destination directory
        if dest_dir is None:
            archive_base_name = os.path.splitext(os.path.basename(archive_path))[0]
            dest_dir = os.path.join('.', f"{archive_base_name}{ColabArchiveManager._DEFAULT_DECOMPRESS_DIR_SUFFIX}")
            print(f"Destination directory not specified. Using default: {dest_dir}")

        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Determine format from file extension
        _, ext = os.path.splitext(archive_path.lower())
        if ext == '.zip':
            archive_format = 'zip'
        elif ext in ['.tar', '.gz', '.tgz', '.bz2', '.tbz', '.tbz2']: # tarfile can handle various extensions
             # Need to check if it's actually a tar file regardless of compression extension
             try:
                 # tarfile.open needs a mode like 'r' or 'r:*'
                 # 'r:*' automatically detects compression
                 with tarfile.open(archive_path, 'r:*') as tar_ref:
                      # Just check if we can read it, don't extract yet
                      tar_ref.getnames() # This will raise exception if not a valid tar file
                 archive_format = 'tar' # It's a tar archive (potentially compressed)
             except tarfile.ReadError:
                  raise ValueError(f"File {archive_path} has a tar-like extension but is not a valid tar archive.")
             except Exception as e:
                  raise RuntimeError(f"Could not open file {archive_path} as tar archive: {e}")

        else:
            raise ValueError(f"Unsupported archive format based on extension: {ext}. Supported: .zip, .tar, .tar.gz, .tar.bz2")


        print(f"Decompressing {archive_path} to {dest_dir}...")

        try:
            if archive_format == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
            elif archive_format == 'tar':
                 # tarfile.open with 'r:*' handles tar, tar.gz, tar.bz2 based on content/extension
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(dest_dir)
            # elif other formats...

            print("Decompression successful.")

        except Exception as e:
            print(f"Error decompressing {archive_path} to {dest_dir}: {e}")
            # Optional: Clean up partially extracted files in dest_dir
            # Consider adding a cleanup step here if decompression fails
            raise RuntimeError(f"Decompression failed: {e}")

        return dest_dir

# --- Hướng dẫn sử dụng Class ---

# Tạo một vài file/thư mục giả để test trong Colab
# !mkdir test_folder
# !echo "This is file1" > test_file1.txt
# !echo "This is file2" > test_folder/test_file2.txt
# !echo "Ignore this file" > ignore_me.log

# --- Nén File/Thư mục ---
print("\n--- Testing Compression ---")

# TH1: Nén toàn bộ thư mục hiện tại (.) với tên và định dạng mặc định (archive_YYYYMMDD_HHMMSS.zip)
# try:
#     archive_path_default = ColabArchiveManager.compress()
#     print(f"Default archive created: {archive_path_default}")
# except Exception as e:
#     print(f"Error testing default compression: {e}")


# TH2: Nén một file hoặc thư mục chỉ định với tên và định dạng mặc định
# try:
#     archive_path_single = ColabArchiveManager.compress(sources='test_folder') # Nén thư mục test_folder
#     # archive_path_single_file = ColabArchiveManager.compress(sources='test_file1.txt') # Nén file test_file1.txt
#     print(f"Single source archive created: {archive_path_single}")
# except Exception as e:
#     print(f"Error testing single source compression: {e}")


# TH3: Nén nhiều file/thư mục với tên và định dạng tùy chỉnh (tar.gz)
# try:
#     sources_list = ['test_file1.txt', 'test_folder']
#     archive_path_custom = ColabArchiveManager.compress(
#         sources=sources_list,
#         archive_name='my_custom_archive',
#         format='tar.gz',
#         exclude='*.log' # Ví dụ: loại trừ các file .log
#     )
#     print(f"Custom archive created: {archive_path_custom}")
# except Exception as e:
#     print(f"Error testing custom compression: {e}")

# # TH4: Nén toàn bộ ngoại trừ một số file/thư mục
# try:
#     archive_path_exclude = ColabArchiveManager.compress(
#         sources='.',
#         archive_name='archive_without_ignores',
#         format='zip',
#         exclude=['*.log', 'test_folder/*'] # Ví dụ: loại trừ file .log và mọi thứ trong test_folder
#     )
#     print(f"Archive with exclude created: {archive_path_exclude}")
# except Exception as e:
#     print(f"Error testing exclude compression: {e}")


# --- Giải nén File ---
print("\n--- Testing Decompression ---")

# Giả sử bạn đã có một file archive, ví dụ 'my_custom_archive.tar.gz' từ bước nén
# archive_to_decompress = 'my_custom_archive.tar.gz'

# TH1: Giải nén với thư mục đích mặc định (tạo thư mục mới tên là my_custom_archive_extracted)
# try:
#     if 'archive_path_custom' in locals() and os.path.exists(archive_path_custom):
#          archive_to_decompress = archive_path_custom
#          decompressed_dir_default = ColabArchiveManager.decompress(archive_to_decompress)
#          print(f"Decompressed to default directory: {decompressed_dir_default}")
#          # Kiểm tra nội dung thư mục (tùy chọn)
#          # !ls -R {decompressed_dir_default}
#     else:
#          print(f"Archive {archive_to_decompress} not found from previous step.")

# except FileNotFoundError as e:
#      print(f"Error testing default decompression: {e}")
# except Exception as e:
#     print(f"Error testing default decompression: {e}")


# TH2: Giải nén với thư mục đích chỉ định
# try:
#     if 'archive_path_custom' in locals() and os.path.exists(archive_path_custom):
#          archive_to_decompress = archive_path_custom
#          custom_dest_dir = 'extracted_here'
#          decompressed_dir_custom = ColabArchiveManager.decompress(archive_to_decompress, dest_dir=custom_dest_dir)
#          print(f"Decompressed to custom directory: {decompressed_dir_custom}")
#          # Kiểm tra nội dung thư mục (tùy chọn)
#          # !ls -R {custom_dest_dir}
#     else:
#          print(f"Archive {archive_to_decompress} not found from previous step.")
# except FileNotFoundError as e:
#      print(f"Error testing custom decompression: {e}")
# except Exception as e:
#     print(f"Error testing custom decompression: {e}")


# --- Dọn dẹp file/thư mục test (Tùy chọn) ---
# !rm -rf test_folder test_file1.txt ignore_me.log *.zip *.tar.gz *.tar.bz2 archive* extracted_here