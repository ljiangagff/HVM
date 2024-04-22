import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse


FIX_HEIGHT = 64
IMAGE_TYPE = 1
SEQ_TYPE = 2


def read_txt(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            res.append(line.rstrip('\n'))
    return res


def json_load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def json_save(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def word_string_to_int(s: str):
    word_list = s.split(" ")
    return [int(i) for i in word_list]


# parse the .npy/.json filename
def parse_filename(s: str):
    filetype: int = 0
    # identity the file is the images file or seqs file
    if s.endswith('.npy'):
        filetype = IMAGE_TYPE
    elif s.endswith('.json'):
        filetype = SEQ_TYPE
    width = s.split('.')[0]
    return (int(width), filetype)


class Loader():
    def __init__(self, image_dir="../im2tex100k/images", label_dir="../im2tex100k/seqs", data_dir="../im2tex100k/data", categorys=['train', 'validate', 'test']) -> None:
        self.categorys = categorys
        self.data_path = {
            'image': image_dir,
            'label': label_dir,
            'load': data_dir
        }
        # define the transform of images
        self.transform = transforms.Compose([
            transforms.Resize(FIX_HEIGHT,),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.9489], std=[0.1565])
        ])
        # vocabularay
        self.vocab_words = read_txt(os.path.join(
            self.data_path['label'], 'vocab.txt'))
        self.vocab = [i for i in range(len(self.vocab_words))]

    def get_info(self, category: str):
        label_path = os.path.join(self.data_path['label'], f'{category}.json')
        return json_load(label_path)

    def load_datas(self, categories: list, device, batch):
        start = time.time()
        datasets = dict()
        for category in categories:
            local_dir = os.path.join(self.data_path['load'], category)
            # the datasets are grouped according to the image width
            # so that the images do not need to pad before training, the key is the width number
            for fn in os.listdir(local_dir):
                (width, filetype) = parse_filename(fn)
                if width not in datasets:
                    datasets[width] = {
                        'id': [],
                        'seq': [],
                    }
                if filetype == IMAGE_TYPE:
                    image_np_array = np.load(os.path.join(local_dir, fn))
                    if 'image' not in datasets[width]:
                        datasets[width]['image'] = image_np_array
                    else:
                        datasets[width]['image'] = np.concatenate(
                            (datasets[width]['image'], image_np_array), axis=0)

                elif filetype == SEQ_TYPE:
                    seqs_info = json_load(os.path.join(local_dir, fn))
                    datasets[width]['id'].extend([i[0] for i in seqs_info])
                    datasets[width]['seq'].extend([i[1] for i in seqs_info])

        t_datasets = dict()
        for k, v in datasets.items():
            inputs = torch.from_numpy(v['image']).to(device)
            labels = [torch.tensor(seq).to(device) for seq in v['seq']]
            padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
            indices = torch.tensor(v['id']).to(device)
            # t_datasets[k] = TensorDataset(inputs, padded_labels, indices)
            d = TensorDataset(inputs, padded_labels, indices)
            t_datasets[k] = DataLoader(d, batch_size=batch, shuffle=True)

        print(
            f'Loaded dataset for {categories}. Loading time {time.time()-start} seconds')
        return t_datasets

    def load_data(self, category: str, device, batch):
        local_dir = os.path.join(self.data_path['load'], category)
        # the datasets are grouped according to the image width
        # so that the images do not need to pad before training, the key is the width number
        start = time.time()
        datasets = dict()
        for fn in os.listdir(local_dir):
            (width, filetype) = parse_filename(fn)
            if width not in datasets:
                datasets[width] = dict()
            if filetype == IMAGE_TYPE:
                # start = time.time()
                # print(f'Begin to load image {fn} of category {category}')
                datasets[width]['image'] = np.load(os.path.join(local_dir, fn))
                # print(f'Loading time {time.time()-start} seconds')
            elif filetype == SEQ_TYPE:
                seqs_info = json_load(os.path.join(local_dir, fn))
                datasets[width]['id'] = [i[0] for i in seqs_info]
                datasets[width]['seq'] = [i[1] for i in seqs_info]
        t_datasets = dict()
        for k, v in datasets.items():
            inputs = torch.from_numpy(v['image']).to(device)
            labels = [torch.tensor(seq).to(device) for seq in v['seq']]
            padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
            indices = torch.tensor(v['id']).to(device)
            # t_datasets[k] = TensorDataset(inputs, padded_labels, indices)
            d = TensorDataset(inputs, padded_labels, indices)
            t_datasets[k] = DataLoader(d, batch_size=batch, shuffle=True)

        print(
            f'Loaded dataset for {category}. Loading time {time.time()-start} seconds')
        return t_datasets

    def group_images(self) -> None:
        outpath = self.data_path['load']
        image_path = self.data_path['image']
        for c in self.categorys:
            label_path = os.path.join(self.data_path['label'], f'{c}.json')
            self.grouped = dict()
            self.ungrouped = json_load(label_path)
            # the grouped seqs and images are in self.grouped
            print(f'Begin to load images for category {c}')
            self.load_image_and_group(image_path)
            prefix = os.path.join(outpath, c)
            num_image = 0
            num_seq = 0
            if not os.path.exists(prefix):
                os.mkdir(prefix)
            for width, v in self.grouped.items():
                image_file_name = os.path.join(prefix, str(width)) + '.npy'
                seq_file_name = os.path.join(prefix, str(width)) + '.json'
                n1 = len(v['images'])
                n2 = len(v['seqs'])
                num_image += n1
                num_seq += n2
                start = time.time()
                imgs = np.array(v['images'])
                # print(f'Convert to npy cost {time.time()-start} seconds, begin to save images for category {c} with shape {imgs.shape}')
                start = time.time()
                np.save(image_file_name, imgs)
                json_save(seq_file_name, v['seqs'])
                print(
                    f'Saved {n1} images {n2} seqs for {c} data with width {width}, saving cost {time.time()-start} seconds')
            print(f'Total images: {num_image}')
            print(f'Total seqs: {num_seq}')

    def load_image_and_group(self, image_dir: str) -> None:
        images = []
        groups = {}
        seqs = self.ungrouped
        for i in range(len(seqs)):
            # each seq has 'name' key, indicating the seq's image filename
            fn = seqs[i]['name']
            image = Image.open(os.path.join(image_dir, fn)).convert('L')
            ni = np.asarray(self.transform(image))
            # Get the image width and group the images by width
            width = ni.shape[2]
            if width not in groups:
                groups[width] = list()
            groups[width].append(i)
            images.append(ni)

        print('Images load complete')
        for width in groups:
            index_list = groups[width]
            self.grouped[width] = {
                "seqs": [(i, word_string_to_int(seqs[i]['seq'])) for i in index_list],
                "images": [images[i] for i in index_list]
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default="../im2tex90k/images")
    parser.add_argument('--seq_dir', type=str, default="../im2tex90k/seqs")
    parser.add_argument('--data_dir', type=str, default="../im2tex90k/data")
    opt = parser.parse_args()

    IMAGE_DIR = opt.image_dir
    LABEL_DIR = opt.seq_dir
    DATA_DIR = opt.data_dir

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ld = Loader(image_dir=IMAGE_DIR, label_dir=LABEL_DIR,
                data_dir=DATA_DIR, categorys=['train', 'test'])
    '''
    Group the image and save to .npy files with .json information
    '''
    print("Begin grouping images")
    ld.group_images()
