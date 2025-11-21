import zipfile
import os
import sys

def zip_folder(folder_path, output_path):
    """
    将一个文件夹（包括其所有内容）压缩到一个zip文件中。
    """
    print(f"开始压缩文件夹: {folder_path}")
    
    # 使用 'w' 模式创建一个新的 zip 文件，ZIP_DEFLATED 表示使用压缩
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # os.walk 会遍历文件夹下的所有文件和子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 构建完整的本地文件路径
                file_path = os.path.join(root, file)
                # 计算在 zip 文件中的相对路径，以保持目录结构
                # os.path.relpath 会得到相对于 folder_path 的路径
                arcname = os.path.relpath(file_path, start=os.path.dirname(folder_path))
                
                # 将文件写入 zip 文件
                zipf.write(file_path, arcname=arcname)
                print(f"已添加: {arcname}")

    print(f"\n压缩完成！文件已保存为: {output_path}")

if __name__ == '__main__':
    # 要压缩的文件夹名称
    folder_to_compress = 'cirr_dataset'
    # 输出的 zip 文件名称
    output_zip_file = 'cirr_dataset.zip'
    
    # 检查文件夹是否存在
    if not os.path.isdir(folder_to_compress):
        print(f"错误：文件夹 '{folder_to_compress}' 不存在。")
        sys.exit(1)
        
    # 调用函数进行压缩
    zip_folder(folder_to_compress, output_zip_file)
