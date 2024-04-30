# Img2latex-HV-attention
This is a repo for implementing the HV attention based mathematical expression recognition model

# Dataset and Checkpoints

You can retrieve the data we have preprocessed and the checkpoints files of HVM-B and HVM-L in this [Google Drive Folder](https://drive.google.com/drive/folders/1NyKZRK_DlUYmAnwMm31kAnGRwlHWlJjI?usp=sharing). Unzip data.zip give you two folders. Run the following code and organize your folder as follows.

```
cd $WORK_DIR
git clone https://github.com/ljiangagff/HVM
mkdir im2tex100k
unzip data.zip 

HVM (source code directory)
im2tex100k (dataset directory)
    |-- data
        |-- train
        |-- validate
        |-- test
    |-- seqs
    |-- log
    |-- checkpoints
```

The log and checkpoints directory are made by you with mkdir command.

# Training

If the source code and data are organized as listed , there is no need to input parameters where all the parameters are set to default. Simply run the following command. You need to assign the spare gpu id for training.

```
cd $WORK_DIR/HVM && python training.py --device=$YOUR_DEVICE_ID
```

Otherwise, you have to assign data_dir, seq_dir, log_dir and model_dir to where you put the data and would liek to save the log file and saved ckpt to. You can also change the hyper parameters shown in train.py.

# Evaluation

Make your folder as follows

```
im2tex100k
|-- test
    |-- ckpt
        |-- HVM-B.pth
        |-- HVM-L.pth
    |-- result
        |-- decoded
        |-- images
        |-- tex
```

Run the following command: 

```
cd $WORK_DIR/HVM && python test.py --device=$YOUR_DEVICE_ID
```

The evaluated results will be reported in the shell, represented by BlEU4, BLEU4_ws  and EMA value. The decoded results are saved im2tex100k/test/decoded. You can further compile the result with render_latex.py and evaluate the result with measure.py.

# Results

| Methods| BLEU4   |  TED| EMA |
| :--------  | :-----:  | :----:  | :----:|
| HVM-B | 92.36 | 0.0504 | 56.12 |
| HVM-B | 93.05 | 0.0447 | 59.63 |
| Nougat-Latex | 91.35 | 0.0576 | 50.43 |

