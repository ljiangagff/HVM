import os, json 
import subprocess
from string import Template
import numpy as np
from PIL import Image

import concurrent.futures

TEST_DIR = "../im2tex100k/test/result/decoded/HVM-B.json"
TRUTH_IMAGE_DIR = "../im2tex100k/test/result/images/truth/"
TRUTH_TEX_DIR = "../im2tex100k/test/result/tex/truth/"

DECODED_DIR = "../im2tex100k/test/result/decoded/"
OUTPUT_DIR = "../im2tex100k/test/result/images/"
TEX_DIR = "../im2tex100k/test/result/tex/"

FILTER_LIST = []

template = Template(r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\begin{document}
\begin{displaymath}
${latex_equation}
\end{displaymath}
\end{document}
""")

# \newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
# \newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
TOTAL_COUNT = 0
SUCCEED = 0

def render_truth():
    with open(TEST_DIR) as f:
        data = json.load(f)
        print(len(data))
        # for d in data:
        #     seq = d['raw'] 
        #     name = d['name'] 
        #     output_an_latex(seq, name, TRUTH_TEX_DIR, TRUTH_IMAGE_DIR)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(output_an_latex, " ".join(d['truth_seq']), d['image_name'], TRUTH_TEX_DIR, TRUTH_IMAGE_DIR) for d in data]


def crop_image(img, output_path, default_size=None):
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            old_im.save(output_path)
            return False
        else:
            assert len(default_size) == 2, default_size
            x_min, y_min, x_max, y_max = 0, 0, default_size[0], default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
            old_im.save(output_path)
            return False
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    old_im.save(output_path)
    return True


def read_latex_strings(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        return data


def render_an_image_pair(item, tex_dir, image_dir):
    image_name = item['image_name']
    decoded = " ".join(item['decoded_seq'])
    # truth = " ".join(item['truth_seq'])
    # output_an_latex(truth, "truth"+image_name, tex_dir, image_dir)
    output_an_latex(decoded, "decoded"+image_name, tex_dir, image_dir)


def output_an_latex(seq_content, filename, tex_dir, image_dir):
    l = seq_content
    latex_name = os.path.join(tex_dir, filename[:-4])
    out_name = os.path.join(image_dir, filename)
    if os.path.exists(out_name):
        return

    tex_filename = latex_name+'.tex'
    log_filename = latex_name+'.log'
    aux_filename = latex_name+'.aux'
    pdf_filename = tex_filename[:-4]+'.pdf'
    png_filename = tex_filename[:-4]+'.png'

    with open(tex_filename, "w") as f:
        tex_content = template.safe_substitute({"latex_equation": l})
        f.write(tex_content)

    _ = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                            f'--output-directory={tex_dir}', tex_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.remove(tex_filename)
    os.remove(log_filename)
    os.remove(aux_filename)

    try:
        os.system(f"convert -density 200 -quality 100 {pdf_filename} {png_filename} > /dev/null 2>&1")
        crop_image(png_filename, out_name)
        os.remove(pdf_filename)
        os.remove(png_filename)
        return 1
    except:
        pass

def ensure_mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def render_result_collection(file):
    data = read_latex_strings(DECODED_DIR+file)
    out_tex_dir = TEX_DIR+file[:-5]
    out_image_dir = OUTPUT_DIR+file[:-5]
    print(f"Compiling latex of {file}")
    ensure_mkdir(out_tex_dir)
    ensure_mkdir(out_image_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(render_an_image_pair, d, out_tex_dir, out_image_dir) for d in data]
    print(f"Rendering {file} completed")


if __name__ == "__main__":
    ensure_mkdir(TRUTH_IMAGE_DIR)
    ensure_mkdir(TRUTH_TEX_DIR)
    render_truth()

    all_files = os.listdir(DECODED_DIR)
    files = []
    for file in all_files:
        if file[:-5] not in FILTER_LIST:
            files.append(file)

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(render_result_collection, file) for file in files]

