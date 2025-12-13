import numpy as np

# 文件路径
file_path = '/Users/gzp/Desktop/exp/DATA/SHREC2017_data/train_data.npy'

try:
    # 使用 mmap_mode='r' 可以只读取元数据而不加载整个文件到内存，速度更快且防爆内存
    data = np.load(file_path, mmap_mode='r')
    print("=" * 30)
    print(f"数据形状 (Shape): {data.shape}")
    print("=" * 30)

    # 打印维度含义提示
    N, C, T, V, M = data.shape
    print(f"N (样本数): {N}")
    print(f"C (通道数): {C}  <-- 关键关注点")
    print(f"T (帧数):   {T}")
    print(f"V (关节点): {V}")
    print(f"M (人数):   {M}")

except FileNotFoundError:
    print(f"错误: 找不到文件，请检查路径是否正确: {file_path}")
except Exception as e:
    print(f"读取发生错误: {e}")