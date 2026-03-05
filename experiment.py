import os
import shutil

def copy_images_based_on_text(source_dir, target_dir, text_file):
    """
    根据文本文件中提供的文件名，将对应的图片从源文件夹复制到目标文件夹。

    :param source_dir: 源文件夹路径，包含所有图片
    :param target_dir: 目标文件夹路径，将选定的图片复制到这里
    :param text_file: 文本文件路径，包含要复制的图片文件名
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    # 读取文本文件中的文件名
    with open(text_file, 'r') as f:
        file_names = f.read().splitlines()

    # 遍历文件名，将对应的图片从源文件夹复制到目标文件夹
    for file_name in file_names:
        source_path = os.path.join(source_dir, file_name+'_pixels0.png')
        target_path = os.path.join(target_dir, file_name+'.png')

        # 检查源文件是否存在
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)  # 使用 copy2 保留元数据
            print(f"Copied {source_path} to {target_path}")
        else:
            print(f"File not found: {source_path}")

# 示例用法
if __name__ == "__main__":
    source_directory = r'/home/x3090/work/dsw/INR/Dataset/NUAA-SIRST/masks'  # 替换为你的源文件夹路径
    target_directory = r'/home/x3090/work/dsw/INR/NUAA/test/masks'  # 替换为你的目标文件夹路径
    text_file_path = r'/home/x3090/work/dsw/INR/Dataset/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt'  # 替换为你的文本文件路径

    copy_images_based_on_text(source_directory, target_directory, text_file_path)