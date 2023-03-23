'''
    Evaluate on first 5 images from dataset and some new images

    Example command:
        python my_canny2image.py lightning_logs/original_sd/inference/control_sd15_ini.txt
        python my_canny2image.py lightning_logs/tgbh_run1/inference/epoch=101-step=4849.txt

    Output folder:
        output/
            original_sd/
                control_sd15_ini/
                    000/
                        input.jpg
                        edge_map.jpg
                        samples.jpg # strip of 4 images
                        args.txt
                    001/
                    ...
                    ira/
                    steve/
                    ...
            fill_run0/
                epoch=1-step=7499/
                ...
            tgbh_run0/
                ...
'''

from share import * # this suppresses verbose mode

import os
import sys
import torch
import einops
import imageio
import numpy as np
from pytorch_lightning import seed_everything

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector


# Helper functions
def inference(model, out_dir, run, img_path, prompt):

    # Set seed and shared configs
    seed = 0
    seed_everything(seed)
    num_samples = 4
    a_prompt = ''
    n_prompt = ''
    guess_mode = False
    res = 512
    low_threshold = 100
    high_threshold = 200
    scale = 9.0
    ddim_steps = 50
    eta = 0.0
    strength = 1.0

    # Prepare inputs for model
    img = imageio.imread(img_path)
    img = resize_image(HWC3(img), res)
    H, W, C = img.shape
    edge_map = apply_canny(img, low_threshold, high_threshold)
    edge_map = HWC3(edge_map)
    control = torch.from_numpy(edge_map).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    # Generate images
    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    results = [x_samples[i] for i in range(num_samples)]
    results = np.concatenate(results, axis=1)

    # Write outputs
    out_dir = os.path.join(out_dir, run)
    os.makedirs(out_dir)
    imageio.imwrite(f'{out_dir}/samples.jpg', results)
    imageio.imwrite(f'{out_dir}/edge_map.jpg', 255 - edge_map)
    imageio.imwrite(f'{out_dir}/input.jpg', img)
    with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
        f.write(f'prompt = {prompt}\n')
        f.write(f'a_prompt = {a_prompt}\n')
        f.write(f'n_prompt = {n_prompt}\n')
        f.write(f'img_path = {img_path}\n')
        f.write(f'seed = {seed}\n')
        f.write(f'num_samples = {num_samples}\n')
        f.write(f'guess_mode = {guess_mode}\n')
        f.write(f'res = {res}\n')
        f.write(f'low_threshold = {low_threshold}\n')
        f.write(f'high_threshold = {high_threshold}\n')
        f.write(f'scale = {scale}\n')
        f.write(f'ddim_steps = {ddim_steps}\n')
        f.write(f'eta = {eta}\n')
        f.write(f'strength = {strength}\n')


# Load original SD and edge detector
process_path = sys.argv[1]
model = create_model('./models/cldm_v15.yaml').cpu()
model_path = 'models/control_sd15_ini.ckpt' 
model.load_state_dict(load_state_dict(model_path, location='cuda'))
if 'original_sd' not in process_path:
    model_path = process_path.replace('inference', 'checkpoints')
    model_path = model_path.replace('.txt', '.ckpt')
    old_dict = load_state_dict(model_path, location='cuda')
    new_dict = {k[14:]:v for k, v in old_dict.items()} # to get rid of the control_model.
    del old_dict
    model.control_model.load_state_dict(new_dict)
model = model.cuda()
ddim_sampler = DDIMSampler(model)
apply_canny = CannyDetector()


# Make output folder
name, exp_name, _, ckpt_name = process_path.split('/')
ckpt_name = ckpt_name.split('.')[0]
out_dir = f'{name}/{exp_name}/output/{ckpt_name}'
os.makedirs(out_dir, exist_ok=True)


# Run processes
with open(process_path, 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.startswith('#'):
        run = line[2:].strip()
        if os.path.exists(os.path.join(out_dir, run)):
            print(f'Skipping {run}')
        else:
            img_path = lines[i+1].strip()
            prompt = lines[i+2].strip()
            inference(model, out_dir, run, img_path, prompt)
            print(f'Finished {run}')
