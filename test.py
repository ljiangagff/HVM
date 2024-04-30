import torch
import os
import argparse
import loader as LD
from val import DecodingVal
from models.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="7")
parser.add_argument('--data_dir', type=str, default="../im2tex100k/data")
parser.add_argument('--label_dir', type=str, default="../im2tex100k/seqs")
parser.add_argument('--model_dir', type=str, default="../im2tex100k/test/ckpt/")
parser.add_argument('--result_dir', type=str, default="../im2tex100k/test/result/decoded/")
parser.add_argument('--log_dir', type=str, default="../im2tex100k/test/log")

opt = parser.parse_args()
DEVICE_ID = int(opt.device)
DATA_DIR = opt.data_dir
LABEL_DIR = opt.label_dir
MODEL_DIR = opt.model_dir
RESULT_DIR = opt.result_dir
LOG_DIR = opt.log_dir

torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device('cuda')


ld = LD.Loader(data_dir=DATA_DIR, label_dir=LABEL_DIR)
my_test = DecodingVal(ld, DEVICE, LOG_DIR, category='test')
my_test.load_data()

def test_models(ckpt_dir, test):
    logger = test.get_logger()
    files = os.listdir(ckpt_dir)
    for file in files:
        if file[-4:] == '.pth':

            my_model = torch.load(os.path.join(ckpt_dir, file), map_location=DEVICE)
            logger.info(f'Begin to validate the model {file}')
            my_model.eval()
            bleu, blue_no, emr, count = test.greedy_validate(my_model, True, result_dir=os.path.join(RESULT_DIR, file[:-4]+".json"))
            logger.critical(f"Validate {count} items. Bleu score: {bleu/count}. Bleu no blank score: {blue_no/count}. Exact match rate: {emr/count}.")
            free_model(my_model)

def free_model(model):
    del model
    torch.cuda.empty_cache()

test_models(MODEL_DIR, my_test)
