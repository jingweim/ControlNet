import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, exp_name, prompt_path, src_dir, tgt_dir):

        # shared attributes
        self.prompts = []
        self.src_paths = []
        self.tgt_paths = []

        if exp_name.startswith('fill'):
            with open(prompt_path, 'rt') as f:
                for line in f:
                    data = json.loads(line)
                    self.prompts.append(data['prompt'])
                    self.src_paths.append(os.path.join(src_dir, data['source']))
                    self.tgt_paths.append(os.path.join(tgt_dir, data['target']))

        elif exp_name.startswith('tgbh'):
            with open(prompt_path, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    self.prompts.append(lines[i+1].strip())
                    fname = (line[2:].strip()).split('/')[-1]
                    self.src_paths.append(os.path.join(src_dir, fname))
                    self.tgt_paths.append(os.path.join(tgt_dir, fname))

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, idx):
        src_path = self.src_paths[idx]
        tgt_path = self.tgt_paths[idx]
        prompt = self.prompts[idx]

        src = cv2.imread(src_path)
        tgt = cv2.imread(tgt_path)

        # Do not forget that OpenCV read images in BGR order.
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)

        # Normalize src images to [0, 1].
        src = src.astype(np.float32) / 255.0

        # Normalize tgt images to [-1, 1].
        tgt = (tgt.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=tgt, txt=prompt, hint=src)

