import torch
import os
import cv2
from PIL import Image
import csv

        
def save_obs_trajectory(obs, path, img_preprocessor=None, dump_video=True):
    if not os.path.isdir(path):
        os.makedirs(path)
    
    if img_preprocessor:
        # Images are decoded in range [-0.5, 0.5], we need to scale them to [0, 255].
        obs = torch.clamp(obs, -0.5, 0.5)
        obs = img_preprocessor.revert_preprocessing(obs)
    
    # Save individual images
    frames = []
    for i in range(obs.shape[0]):
        save_path = path + str(i) + ".png"
        o = obs[i].permute(1, 2, 0).numpy()
        frames.append(Image.fromarray(o, "RGB"))
        # cv2 uses BGR format.
        cv2.imwrite(save_path, cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
    
    if dump_video:
        # Save corresponding gif
        gif_path = os.path.join(path, 'trajectory.gif')
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0)

        
def save_to_csv(data, header_names, path, file_name):
    seq_len = len(data[0]) # assuming all data items have the same length
    with open(path + "/" + file_name, "w", newline="") as f:
        writer = csv.writer(f)
        
        header = ["step"] + header_names
        writer.writerow(header)
        
        for step in range(seq_len):
            row = [step] + [d[step].item() for d in data]
            writer.writerow(row)