import pandas as pd
import json, os
import standardization
import argparse

PAD_WORD = '[PAD]'
START_WORD = '[SOS]'
END_WORD = '[EOS]'
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
MAX_LENGTH = 180
CAT_TRAIN = 'train'
CAT_VALIDATE = 'validate'
CAT_TEST = 'test'


parser = argparse.ArgumentParser()
parser.add_argument('--seq_dir', type=str, default="../../im2tex100k/seqs")
parser.add_argument('--image_dir', type=str, default="../../im2tex100k/images")
opt = parser.parse_args()

SEQ_DIR = opt.seq_dir
IMAGE_DIR = opt.image_dir
DATA_PATHS = {
    'formula': f'{SEQ_DIR}/formulas.norm.lst',
    'train': f'{SEQ_DIR}/train_filter.lst',
    # 'validate': f'{SEQ_DIR}/validate_filter.lst',
    'test': f'{SEQ_DIR}/test_filter.lst',
    'vocab': f'{SEQ_DIR}/latex_vocab.txt',
}


class DataFusion():
    # def __init__(self, data_paths=[], data_categorys=[CAT_TRAIN, CAT_VALIDATE, CAT_TEST]):
    def __init__(self, data_paths=[], data_categorys=[CAT_TRAIN, CAT_TEST]):
        self.categorys = data_categorys
        self.paths = data_paths
        self.vocab = [PAD_WORD, START_WORD, END_WORD]
        self.data = dict()

    def read_data(self):
        # read the vocab file
        with open(self.paths['vocab'], "r") as f:
            # start from 3 because we have pad, start and end token at beginning
            for voc in f:
                self.vocab.append(voc.rstrip('\n'))

        with open(self.paths['formula'], "r") as f:
            self.data['seqs'] = list()
            for line in f:
                seq_string = line.rstrip('\n')
                try:
                    seq_list = seq_string.split(" ")
                    seq_tokenized = [START_TOKEN]
                    seq_tokenized += [self.vocab.index(s) for s in seq_list]
                    seq_tokenized.append(END_TOKEN)
                    seq_tokenized = standardization.clean_seq(
                        seq_tokenized, filter_blank=False)
                    form = {
                        "valid": True,
                        "name": "",
                        "len": len(seq_tokenized),
                        "seq": " ".join([str(i) for i in seq_tokenized]),
                        "raw": seq_string
                    }
                    if form['len'] > MAX_LENGTH+2:
                        form['valid'] = False
                    self.data['seqs'].append(form)
                except Exception:
                    # print(f'Invalid vocab found, skip this seq')
                    self.data['seqs'].append({
                        "valid": False,
                        "error": True,
                        "raw": seq_string,
                    })

        # read the train, validate and test file
        for c in self.categorys:
            self.data[c] = list()
            with open(self.paths[c], "r") as f:
                for line in f:
                    (filename, index) = line.split(" ")
                    form = self.data['seqs'][int(index)]
                    # form cannot be False which is a bad seq that includes invalid vocab
                    if form['valid'] and os.path.exists(os.path.join(IMAGE_DIR, filename)):
                        form['name'] = filename
                        self.data[c].append(form)
                    elif 'error' in form:
                        pass
                        # print(
                        #     f'Invalid vocab found for seq image {filename}, skip this seq')
                    else:
                        pass
            # get the distribution according to the length of the seqs
            self.get_distribution([i['len'] for i in self.data[c]])

    def write_data(self):
        # write the train, validate and test files
        for c in self.categorys:
            forms = self.data[c]
            with open(f'{SEQ_DIR}/{c}.json', 'w') as f:
                json.dump(forms, f, indent=4, ensure_ascii=False)
            print(
                f'Complete Seq Info export for {c}, begin to category the images')
        # write the vocab list
        with open(f'{SEQ_DIR}/vocab.txt', 'w') as f:
            for voc in self.vocab:
                f.write(voc+'\n')

    def get_distribution(self, dis: list):
        bins = [0, 64, 160, 256, 512, 1024]
        bins = [i+2 for i in bins]
        distribution = pd.cut(dis, bins)
        print(f'{type} data length distribition: \n{pd.value_counts(distribution)}')


if __name__ == "__main__":
    fusion = DataFusion(data_paths=DATA_PATHS)
    fusion.read_data()
    fusion.write_data()
