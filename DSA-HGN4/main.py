import argparse
import sys
import os
import shutil
import zipfile
import time
import random
import numpy as np
import torch

# torchlight
import torchlight
from torchlight import import_class

# ✅ 修复：直接在这里定义 init_seed，防止 import 失败导致的 NameError
def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':
    # ✅ 修复：直接调用上面定义的函数，不再需要 try-except
    init_seed(0)

    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()

    try:
        processors['finetune_evaluation'] = import_class('processor.recognition.FT_Processor')
    except Exception as e:
        print(f"[Warning] Failed to load 'finetune_evaluation': {e}")

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    if arg.processor not in processors:
        print(f"Error: Processor '{arg.processor}' not found. Available: {list(processors.keys())}")
        sys.exit(1)

    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    if p.arg.phase == 'train':
        # save src
        save_src(p.arg.work_dir)

    p.start()