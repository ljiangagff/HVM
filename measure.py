from render_latex import read_latex_strings
import concurrent.futures
import os, json, cv2
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from image_similarity_measures.quality_metrics import rmse, ssim
from preprocessing.standardization import filter_blank, clean_seq
import Levenshtein

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="im2tex100k")
opt = parser.parse_args()
DATASET_NAME = opt.dataset

IMAGE_DIR = f"../{DATASET_NAME}/test/result/images/"
DECODED_DIR = f"../{DATASET_NAME}/test/result/decoded/"
RESULT_DIR = f"../{DATASET_NAME}/test/measure.json"


def bleu(tf_tokens, decoded_tokens, weights=(0.25, 0.25, 0.25, 0.25)):
    bleu4_value = sentence_bleu([tf_tokens], decoded_tokens, weights=weights)
    return float(bleu4_value)


class MetricComparison:
    def __init__(self, result_path) -> None:
        self.data = read_latex_strings(os.path.join(DECODED_DIR, result_path))
        self.image_dir = IMAGE_DIR+result_path[:-5]
        self.ssim = 0
        self.rmse = 0

        self.bleu4_total = 0.0
        self.bleu4_ws = 0.0
        self.bleu1_total = 0.0
        self.edit_distacne = 0.0
        # self.image_edit_distance = 0.0
        self.exact_match = 0
        self.exact_match_ws = 0
        self.valid = 0
        self.count = 0
        self.total = 0
        self.out_weight = 1

    def main(self):
        # for d in self.data:
        #     self.compare(d)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for d in self.data:
                executor.submit(self.compare, d)

    def compare(self, item):
        self.total += 1
        truth_token = [int(i) for i in item['truth_token']]
        decoded_token = item['decoded_token']
        if self.out_weight:
            truth_token = clean_seq(truth_token)
            decoded_token= clean_seq(decoded_token)
        
        if truth_token == decoded_token:
            bleu4_value = 1
            bleu1_value = 1
            bleu4_ws = 1
            edit_distacne = 0
            self.exact_match += 1
            self.exact_match_ws += 1
        else:
            bleu4_value = bleu(truth_token, decoded_token)
            bleu1_value = bleu(truth_token, decoded_token,
                               weights=(1.0, 0, 0, 0))
            truth_ws_blank = filter_blank(truth_token)
            decoded_ws_blank = filter_blank(decoded_token)
            if truth_ws_blank == decoded_ws_blank:
                self.exact_match_ws += 1
            bleu4_ws = bleu(truth_ws_blank, decoded_ws_blank)
            s1 = "".join([chr(i) for i in truth_token])
            s2 = "".join([chr(i) for i in decoded_token])
            edit_distacne = Levenshtein.distance(s1, s2)/max(len(s1), len(s2))

        self.bleu4_total += bleu4_value
        self.bleu4_ws += bleu4_ws
        self.bleu1_total += bleu1_value
        self.edit_distacne += edit_distacne

        image_name = item['image_name']
        truth_filename = f"{IMAGE_DIR}truth/{image_name}"
        decoded_filename = os.path.join(self.image_dir, f"decoded{image_name}")
        # print(f"{truth_filename}")

        if not os.path.exists(truth_filename):
            # print(f"{truth_filename} is lost")
            # The truth of this test item can not be compiled is Illegal
            # print(f"Illegal rendering for {truth_filename}")
            return

        # There's decoded image, compare images
        # print(decoded_filename)
        self.count += 1
        if os.path.exists(decoded_filename):
            img1 = cv2.imread(truth_filename, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(decoded_filename, cv2.IMREAD_GRAYSCALE)

            height = max(img1.shape[0], img2.shape[0])
            width = max(img1.shape[1], img2.shape[1])

            pad_1 = ((0, height-img1.shape[0]), (0, width-img1.shape[1]))
            pad_2 = ((0, height-img2.shape[0]), (0, width-img2.shape[1]))
            img1 = np.pad(img1, pad_1, mode="constant", constant_values=255)
            img2 = np.pad(img2, pad_2, mode="constant", constant_values=255)
            # d, l, _, _ = img_edit_distance(img1, img2)
            img1 = np.expand_dims(img1, axis=-1)
            img2 = np.expand_dims(img2, axis=-1)
            sim_ssim = ssim(img1, img2)
            sim_rmse = rmse(img1, img2)
            # print(f"The ssim image similarity for {image_name} is {sim}")
            self.ssim += sim_ssim
            self.rmse += sim_rmse
            self.valid += 1
            # self.image_edit_distance += d/l


def measure(dir):
    obj = MetricComparison(dir)
    obj.out_weight = True
    obj.main()
    result = {
        "bleu4": obj.bleu4_total/obj.total,
        "bleu4_ws": obj.bleu4_ws/obj.total,
        "edit_distacne": obj.edit_distacne/obj.total,
        "ema": obj.exact_match/obj.total,
        "ema_ws": obj.exact_match_ws/obj.total,
        "ssim": obj.ssim/obj.count,
        "rmse": obj.rmse/obj.valid,
        "unrendered": 1-obj.valid/obj.count,
    }
    print(result)
    return result


if __name__ == "__main__":
    measurements = dict()
    results = os.listdir(DECODED_DIR)
    for r in results:
        print(f"Begin measuring {r}")
        measurements[r] = measure(r)
    with open(RESULT_DIR, "w") as f:
        json.dump(measurements, f, indent=4)
