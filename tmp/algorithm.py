from nougat import NougatModel
from nougat.utils.device import move_to_device
from PIL import Image
import numpy as np
import os
import json
import concurrent.futures


TEST_DIR = "/workspace/FR/im2tex100k/seqs/test.json"
IMAGE_DIR = "/workspace/FR/im2tex100k/test/result/images/truth/"
RESULT_DIR = "/workspace/FR/im2tex100k/test/result/nougat.json"


class NougatTest:
    def __init__(self):
        self.image_map = {}
        self.model = NougatModel.from_pretrained("/workspace/nougat/1/nougat-0.1.0-base")
        # self.model = move_to_device(self.model,  bf16=True, cuda=True)
        self.model.to("cuda:1")
        self.model.eval()

    def nougat(self, d):
        name = d['name']
        image_file = os.path.join(IMAGE_DIR, name)
        image = Image.open(image_file).convert("RGB")
        latex = self.model.inference(image, early_stopping=False)['predictions'][0]
        print(f"{name} output: {latex}")
        self.image_map[name] = latex

    def run_test(self):
        with open(TEST_DIR) as f:
            data = json.load(f)

        for d in data:
            self.nougat(d)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        #     for d in data:
        #         executor.submit(self.nougat, d)
        with open(RESULT_DIR, "w") as f:
            json.dump(self.image_map, f)


if __name__ == "__main__":
    t = NougatTest()
    t.run_test()
