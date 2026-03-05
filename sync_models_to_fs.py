import os
import shutil

src_base = '/root/autodl-tmp/models'
dest_base = '/root/autodl-fs/models'

if not os.path.exists(dest_base):
    os.makedirs(dest_base)

def count_items(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        count += len(files) + len(dirs)
    return count

for item in sorted(os.listdir(src_base)):
    src_path = os.path.join(src_base, item)
    dest_path = os.path.join(dest_base, item)

    if os.path.isdir(src_path):
        if os.path.exists(dest_path):
            src_count = count_items(src_path)
            dest_count = count_items(dest_path)
            
            if src_count == dest_count:
                print(f"⏭️ 跳过文件夹 {item} (子文件数量一致: {src_count})")
                continue
            else:
                print(f"🔄 合并/覆盖文件夹 {item} (数量不一致: 源={src_count}, 目标={dest_count})")
                # 使用 copytree 的 dirs_exist_ok 避免 rmtree 可能出现的锁定问题
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            print(f"📁 复制新文件夹 {item}")
            shutil.copytree(src_path, dest_path)
    else:
        # 对根目录下的文件(如csv)通过大小比对
        if os.path.exists(dest_path):
            src_size = os.path.getsize(src_path)
            dest_size = os.path.getsize(dest_path)
            if src_size == dest_size:
                continue
            else:
                print(f"📄 覆盖文件 {item}")
        else:
            print(f"📄 复制新文件 {item}")
        shutil.copy2(src_path, dest_path)

print("✅ 同步完成！")
