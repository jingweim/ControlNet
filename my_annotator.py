import cv2
import os
from tqdm import tqdm
from gradio_annotator import canny, hed, mlsd

# global parameters
res = 512
l, h = 100, 200             # canny
thr_v, thr_d = 0.1, 0.1     # mlsd
method = 'mlsd'             # 'canny', 'hed', 'mlsd'

# output folder
out_dir = f'input/tgbh/{method}'
if method == 'canny':
    out_dir += f'_{l}_{h}'
elif method == 'mlsd':
    out_dir += f'_{thr_v}_{thr_d}'
os.makedirs(out_dir, exist_ok=True)

# edge extraction
img_folder = '../diffusers/examples/dreambooth/data/tgbh/images'
img_fnames = sorted(os.listdir(img_folder))
for img_fname in tqdm(img_fnames):
    img_path = os.path.join(img_folder, img_fname)
    img = cv2.imread(img_path)
    if method == 'canny':
        edge = canny(img, res, l, h)[0]
    elif method == 'hed':
        edge = hed(img, res)[0]
    elif method == 'mlsd':
        edge = mlsd(img, res, thr_v, thr_d)[0]
    cv2.imwrite(os.path.join(out_dir, img_fname), edge)