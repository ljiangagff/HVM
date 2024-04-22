from render_latex import read_latex_strings
import concurrent.futures
import os, json
from nltk.translate.bleu_score import sentence_bleu
from preprocessing.standardization import filter_blank
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="im2tex100k")
opt = parser.parse_args()
DATASET_NAME = opt.dataset

DECODED_DIR = f"../{DATASET_NAME}/test/result/compare_grid/"
RESULT_DIR = f"../{DATASET_NAME}/test/grid_compare.json"

def bleu(tf_tokens, decoded_tokens, weights=(0.25, 0.25, 0.25, 0.25)):
    bleu4_value = sentence_bleu([tf_tokens], decoded_tokens, weights=weights)
    return float(bleu4_value)


class MetricComparison:
    def __init__(self, result_path) -> None:
        self.data = read_latex_strings(os.path.join(DECODED_DIR, result_path))
        self.total = [0, 0, 0, 0, 0]
        self.exact_match = [0, 0, 0, 0, 0]
        self.exact_match_ws = [0, 0, 0, 0, 0]
        self.bleu4_total = [0, 0, 0, 0, 0]
        self.bleu4_total_ws = [0, 0, 0, 0, 0]

    def main(self): 
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor: 
            for d in self.data:
                executor.submit(self.compare, d)

    def compare(self, item):
        truth_token = item['truth_token']
        decoded_token = item['decoded_token']
        truth_ws_blank = filter_blank(truth_token)
        decoded_ws_blank = filter_blank(decoded_token)
        l = len(truth_token)
        index = math.floor(l/32)
        if l >= 4:
            l = 4

        if truth_token == decoded_token:
            bleu4_value = 1
            exact_match = 1
        else:
            bleu4_value = bleu(truth_token, decoded_token)
            exact_match = 0

        if truth_ws_blank == decoded_ws_blank:
            exact_match_ws = 1
            bleu4_value_ws = 1
        else:
            exact_match_ws = 0
            bleu4_value_ws = bleu(truth_ws_blank, decoded_ws_blank)


        self.total[index] += 1
        self.bleu4_total[index] += bleu4_value
        self.bleu4_total_ws[index] +=bleu4_value_ws 
        self.exact_match[index] += exact_match
        self.exact_match_ws[index] += exact_match_ws


def measure(dir):
    obj = MetricComparison(dir)
    obj.main()
    print(obj.total)
    print(obj.exact_match)
    result = {
        "bleu4": [obj.bleu4_total[i]/obj.total[i] for i in range(5)],
        "bleu4_ws": [obj.bleu4_total_ws[i]/obj.total[i] for i in range(5)],
        "ema": [obj.exact_match[i]/obj.total[i] for i in range(5)],
        "ema_ws": [obj.exact_match_ws[i]/obj.total[i] for i in range(5)],
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
