import torch
import time
import os
import argparse
from loader import Loader
from models.model import Model
from val import DecodingVal
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="../im2tex100k/data")
parser.add_argument('--label_dir', type=str, default="../im2tex100k/seqs")
parser.add_argument('--model_dir', type=str, default="../im2tex100k/checkpoints")
parser.add_argument('--log_dir', type=str, default="../im2tex100k/log")

parser.add_argument('--device', type=str, default="0")
parser.add_argument('--scale', type=str, default="256")
parser.add_argument('--batch_size', type=str, default="64")
parser.add_argument('--lr', type=str, default="2e-4")
parser.add_argument('--gamma', type=str, default="0.8")
parser.add_argument('--contrast', type=str, default="0.3")
parser.add_argument('--encoder', type=str, default="0")
parser.add_argument('--text_encoder', type=str, default="0")
parser.add_argument('--decoder', type=str, default="0")
parser.add_argument('--alpha', type=str, default="0.7")
parser.add_argument('--full_attn', type=str, default="0")

parser.add_argument('--prefix', type=str, default="")


opt = parser.parse_args()
DATA_DIR = opt.data_dir
LABEL_DIR = opt.label_dir
MODEL_DIR = opt.model_dir
LOG_DIR = opt.log_dir

PREFIX = opt.prefix
BATCH_SIZE = int(opt.batch_size)
DEVICE_ID = int(opt.device)
HDIM = int(opt.scale)
LR = float(opt.lr)
GAMMA = float(opt.gamma)

CONTRASTIVE_LOSS_WEIGHT = float(opt.contrast)
ALPHA = float(opt.alpha)
FULL_ATTN = int(opt.full_attn)
ENCODER_TYPE = int(opt.encoder)
TEXT_ENCODER_TYPE = int(opt.text_encoder)
DECODER_TYPE = int(opt.decoder)


param_str = f'M{PREFIX}_scale{HDIM}_ie{ENCODER_TYPE}_te{TEXT_ENCODER_TYPE}_d{DECODER_TYPE}_batch{BATCH_SIZE}_contrast{CONTRASTIVE_LOSS_WEIGHT}_alpha{ALPHA}_fa{FULL_ATTN}_lr{LR}_gamma{GAMMA}'
MODEL_DIR = f'{MODEL_DIR}/{param_str}'
LOG_DIR = f'{LOG_DIR}/{param_str}.txt'


# set cuda device
torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device('cuda')

SAVE_ROUND = 100
PATIENCE = 30 


class FR_Training:
    def __init__(self, model, val, trainset, lr) -> None:
        self.model = model
        self.val = val
        self.trainset = trainset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[
                                        20*(i+1) for i in range(14)], gamma=GAMMA, last_epoch=-1)
        self.current_epoch = 0
        self.best_bleu = 0.0
        self.best_epoch = 0
        self.current_best_model_path = ''
        self.current_best_model_state_path = ''
        self.patience = PATIENCE
        self.INTERVAL_TEST = 100
        self.MAX_EPOCH = 300
        self.logger = val.get_logger()
        self.logger.info(f'Learning rate {lr}')

    def epoch_train(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        for epoch in tqdm(range(self.MAX_EPOCH), desc='Epochs'):
            self.total_loss = 0
            self.num_item_loss = 0

            current_lr = self.lr_scheduler.get_last_lr()
            print('\n')
            self.logger.info(
                f'Epoch {epoch} training start, current learning rate {current_lr}')
            self.current_epoch = epoch

            '''
            A training epoch
            '''
            epoch_start = time.time()
            self.training_process()
            self.logger.info(
                f'Training time for epoch {self.current_epoch} {time.time()-epoch_start} seconds, epoch loss {self.total_loss/self.num_item_loss}')
            '''
            Validate process
            '''
            result = self.validate()
            if result:
                self.lr_scheduler.step()
            else:
                self.logger.info(
                    f'Successful training at {self.best_epoch} epochs, bleu_score {self.best_bleu}')
                return

    def training_process(self):
        for k, v in self.trainset.items():
            pbar = tqdm(enumerate(v), desc='Batches', leave=False)
            for num_step, data in pbar:
                self.model.train()
                inputs, labels, indices = data
                teach_forcing_tokens = labels.clone()
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                loss = self.model(inputs, teach_forcing_tokens)
                loss.backward()
                self.optimizer.step()
                # compute loss and display
                pbar.set_postfix({"loss": loss})
                self.total_loss += loss*inputs.shape[0]
                self.num_item_loss += inputs.shape[0]

                # Validate a single formula in random
                # if num_step >= 0 and num_step % self.INTERVAL_TEST == 0:  # try inference to check that it works well
                #     self.model.eval()
                #     with torch.no_grad():
                #         self.val.single_validate(self.model, self.current_epoch)

    def validate(self):
        if self.current_epoch > SAVE_ROUND:
            self.logger.info(
                f'Begin to validate at {self.current_epoch} epoch')
            self.model.eval()
            with torch.no_grad():
                bleu, blue_no, emr, count = self.val.greedy_validate(
                    self.model)
                self.logger.critical(
                    f"Validate {count} items. Bleu score: {bleu/count}. Bleu no blank score: {blue_no/count}. Exact match rate: {emr/count}.")

            current_bleu = bleu/count
            if current_bleu > self.best_bleu:
                self.best_bleu = current_bleu
                self.best_epoch = self.current_epoch
                self.logger.info(
                    f'New best validate at epoch {self.best_epoch}, bleu_score {self.best_bleu}')
                # Delete the old best
                if os.path.exists(self.current_best_model_state_path):
                    os.remove(self.current_best_model_state_path)

                # Save the new best
                new_best_name = f'epoch_{self.current_epoch}_best_{current_bleu}.pth'
                self.current_best_model_path = os.path.join(
                    MODEL_DIR, new_best_name)
                self.current_best_model_state_path = os.path.join(
                    MODEL_DIR, f'state_{new_best_name}')

                torch.save(self.model, self.current_best_model_path)
                torch.save(self.model.state_dict(),
                           self.current_best_model_state_path)

            if self.current_epoch - self.best_epoch > self.patience:
                final_path = os.path.join(
                    MODEL_DIR, f'epoch_{self.current_epoch}_final{current_bleu}.pth')
                torch.save(self.model.state_dict(), final_path)
                return False
        return True


ld = Loader(data_dir=DATA_DIR, label_dir=LABEL_DIR)
my_model = Model(ld.vocab,
                 device=DEVICE,
                 hdim=HDIM,
                 contrastive_loss_weight=CONTRASTIVE_LOSS_WEIGHT,
                 encoder_type=ENCODER_TYPE,
                 decoder_type=DECODER_TYPE,
                 text_encoder_type=TEXT_ENCODER_TYPE,
                 encoder_params={
                     'alpha': ALPHA,
                     'full_attn': True if FULL_ATTN else False,
                 }).to(DEVICE)

my_validate = DecodingVal(ld, DEVICE, LOG_DIR, category='validate')
my_validate.load_data()
my_trainset = ld.load_data('train', DEVICE, BATCH_SIZE)
trainer = FR_Training(my_model, my_validate, my_trainset, LR)
trainer.epoch_train()
