import torch
import os, cv2
import numpy as np
import argparse
import loader as LD
from val import DecodingVal, get_decoded
from models.model import Model

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="7")
opt = parser.parse_args()
DEVICE_ID = int(opt.device)

torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device('cuda')

DATA_DIR = '../im2tex100k/data_tiny'
LABEL_DIR = '../im2tex100k/seqs'
MODEL_DIR = '../im2tex100k/HVM-B.pth'
LOG_DIR = '../im2tex100k/test/log'
RESULT_DIR = "../im2tex100k/test/result/decoded/"

IMAGE_DIR = "../im2tex100k/images/"
IMAGE_OUT_DIR = "../im2tex100k/images_out/"
IMAGE_NAME = "a8ec0c091c.png"

def weight_map_to_rgb(weight_map):
    # Normalize weights to [0, 1]
    weight_map = cv2.normalize(weight_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Apply colormap
    cmap = plt.get_cmap('jet')  # Choose your colormap
    norm = Normalize(vmin=0, vmax=1)  # Normalize weights to [0, 1]
    rgba_img = cmap(norm(weight_map))  # Apply colormap
    # Convert RGBA image to RGB
    rgb_img = np.delete(rgba_img, 3, 2)  # Remove alpha channel
    # Convert to uint8
    rgb_img = (rgb_img * 255).astype(np.uint8)

    return rgb_img


def generate_heatmap(rgb_image, attn_weights, type="h"):
    # Resize the attention weights to match the size of the original image
    # (c, h, w)
    attn_weights = attn_weights.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    awr = cv2.resize(attn_weights, (rgb_image.shape[1], rgb_image.shape[0]))
    awr = np.sum(awr, axis=2)
    heatmap = cv2.normalize(awr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Normalize attention weights to [0, 1]
    # awr = cv2.normalize(awr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  #CAM resize match input image size
    result = heatmap * 0.3 + rgb_image* 0.7
                        

    # plt.imshow(heatmap, cmap='hot')
    # plt.colorbar()  # 添加颜色条
    # plt.axis('off')  # 关闭坐标轴
    # plt.savefig(os.path.join(IMAGE_OUT_DIR, f"{type}_attn_{IMAGE_NAME}.png"))

    # Create an empty heatmap with the same size as the original image
    # heatmap = np.zeros_like(rgb_image, dtype=np.float32)
    # Combine attention weights with each channel of the original image to create heatmap


    # Normalize heatmap to [0, 255] and convert to uint8jkK
    cv2.imwrite(os.path.join(IMAGE_OUT_DIR, f"{type}_attn_{IMAGE_NAME}"), result)

    return heatmap

ld = LD.Loader(data_dir=DATA_DIR, label_dir=LABEL_DIR)
my_model = Model(ld.vocab,
                 device=DEVICE,
                 hdim=256,
                 contrastive_loss_weight=0.3,
                 encoder_type=0,
                 decoder_type=0,
                 text_encoder_type=0,
                 encoder_params={
                     'alpha': 0.7,
                     'full_attn': False,
                 }).to(DEVICE)

my_model.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
my_model.eval()

my_test = DecodingVal(ld, DEVICE, LOG_DIR, category='test')
my_test.load_data()


def infer_attn(model, input):
    input = model(input)
    cvt_last = model.stage[-1]
    ha = cvt_last.cvt_attn['ha'].permute(2, 3, 0, 1)
    va = cvt_last.cvt_attn['va'].permute(2, 3, 0, 1)
    attn = cvt_last.cvt_attn['attn']
    return ha, va, attn
    pass



for width, val in my_test.dataset.items():
    for inputs, labels, indices in iter(val):
        index = indices[0]
        info = my_test.datainfo[index]
        image_name = info['name']
        if image_name == IMAGE_NAME:
            ha, va, attn = infer_attn(my_model.image_encoder, inputs)
            rgb_image = cv2.imread(os.path.join(IMAGE_DIR, IMAGE_NAME))
            heatmap = generate_heatmap(rgb_image, ha, type="h")
            heatmap = generate_heatmap(rgb_image, va, type="v")
            heatmap = generate_heatmap(rgb_image, attn, type="all")
            # prediction = my_model.greedy_search_batch(input)[0].tolist()
            # decoded_seq = get_decoded(prediction)[1:-1]
            # decoded_str_list = my_test.decode_to_latex(decoded_seq)
            # print(" ".join(decoded_str_list))

# Generate heatmap
# Save the heatmap image

