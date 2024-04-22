import random, json
import logging, time
import colorlog
from nltk.translate.bleu_score import sentence_bleu
from preprocessing.standardization import filter_blank

START_TOKEN = 1
END_TOKEN = 2
# INFER_BATCH_SIZE = 64 
INFER_BATCH_SIZE = 64


class DecodingVal():
    def __init__(self, loader, device, log_dir, category='validate'):
        # formatted, unformated and index
        self.device = device
        self.loader = loader
        self.vocab_words = loader.vocab_words
        self.logger = init_logger(log_dir)
        self.std_logger = init_logger('NoFile')
        self.category = category
        pass

    def load_data(self):
        self.dataset = self.loader.load_data(
            self.category, self.device, INFER_BATCH_SIZE)
        self.datainfo = self.loader.get_info(self.category)

    def decode_to_latex(self, decoded_tokens=[]):
        return [self.vocab_words[i] for i in decoded_tokens]

    # def beam_validate(self, val_model, top_k=False):
    #     bleu_total = 0.0
    #     bleu_num = 0
    #
    #     for width, val in self.dataset.items():
    #         for inputs, labels, indices in iter(val):
    #             infos = [self.datainfo[i] for i in indices]
    #             inputs, _ = val_model.image_encoder(inputs)
    #             predictions, _ = beam_search(val_model, inputs)
    #             predictions.detach().cpu()
    #             for i in range(len(infos)):
    #                 info = infos[i]
    #                 truth_seq = self.decode_to_latex(
    #                     [int(i) for i in info['seq'].split(" ")])
    #                 precs = predictions[i] if top_k else predictions[i][:1]
    #                 for d in precs:
    #                     decoded_seq = get_decoded(list(d))
    #                     bleu = compute_bleu(truth_seq, decoded_seq)
    #                     bleu_total += bleu
    #                     bleu_num += 1
    #
    #             self.logger.info(
    #                 f'Current validate items {bleu_num}, bleu_score {bleu_total/bleu_num}')
    #     return bleu_total, bleu_num

    def greedy_validate(self, val_model, is_test=False, result_dir=""):
        bleu_total = 0.0
        bleu_total_no_blank = 0.0
        exact_match = 0
        bleu_num = 0
        # bad_cases = dict()
        result_list = []

        s = time.time()

        for width, val in self.dataset.items():
            for inputs, labels, indices in iter(val):
                inputs, _ = val_model.image_encoder(inputs)
                predictions = val_model.greedy_search_batch(inputs)
                predictions.detach().cpu()
                for i in range(len(indices)):
                    index = indices[i]
                    if index >= len(self.datainfo):
                        missing_seq = labels[i]
                        print(f'A blank test item found for seq: {missing_seq}')
                        continue

                    info = self.datainfo[index]
                    image_name = info['name']

                    # Predict the latex sequence tokens
                    prediction = predictions[i].tolist()
                    decoded_seq = get_decoded(prediction)[1:-1]
                    truth_seq = [int(i) for i in info['seq'].split(" ")][1:-1]

                    if is_test:
                        truth_str_list = self.decode_to_latex(truth_seq)
                        decoded_str_list = self.decode_to_latex(decoded_seq)
                        result_list.append({
                            "image_name": image_name,
                            "truth_seq": truth_str_list,
                            "decoded_seq": decoded_str_list,
                            "truth_token": truth_seq,
                            "decoded_token":decoded_seq 
                        })

                    # if is_test:
                    #     decoded_seq = clean_seq(decoded_seq, filter_blank=False)
                    #     truth_seq = clean_seq(truth_seq, filter_blank=False)

                    if decoded_seq == truth_seq:
                        bleu = 1.0
                        exact_match += 1
                    else:
                        bleu = compute_bleu(truth_seq, decoded_seq)
                        # if bleu <= 0.2:
                        #     print(" ".join(truth_str_list))
                        #     print(" ".join(decoded_str_list))
                        #     print(bleu)

                    # Bleu without blank
                    bleu_total += bleu
                    decoded_ws = filter_blank(decoded_seq)
                    truth_ws= filter_blank(truth_seq)
                    bleu_no_blank = compute_bleu(truth_ws, decoded_ws)
                    bleu_total_no_blank += bleu_no_blank
                    bleu_num += 1

                    # truth_str = " ".join(self.decode_to_latex(truth_seq))
                    # decoded_str = " ".join(self.decode_to_latex(decoded_seq))
                    # if is_test:
                    #     if bleu == 0.0:
                    #         self.logger.critical(f'\nWorst case occured for image {image_name}')
                    #         self.logger.info(truth_str)
                    #         self.logger.info(decoded_str)
                    #         self.logger.info('Bleu 0.0\n')
                    #         continue
                    #     elif bleu <= 0.7:
                    #         bad_cases[image_name] = bleu
                    #         self.logger.info(f'\nBad Case occured for image {image_name}')
                    #         self.logger.info(truth_str)
                    #         self.logger.info(decoded_str)
                    #         self.logger.info(f'Bleu value {bleu} \n')

        # print(f'Very bad case items {len(bad_cases.keys())}')
        # with open("bad_case0.json", "w") as f:
        #     json.dump(bad_cases, f, indent=4)

        # if is_test:
        #     with open(result_dir, "w") as out_result:
        #         json.dump(result_list, out_result)
        #         print(f"Output result file {result_dir}")
        print(time.time()-s)
        return bleu_total, bleu_total_no_blank, exact_match, bleu_num

    # predict is the function of decoder model

    # def single_validate(self, val_model, epoch):
    #     rand_width = random.choice(list(self.dataset.keys()))
    #     inputs, _, indices = next(iter(self.dataset[rand_width]))
    #     rand_index = random.randint(0, inputs.shape[0]-1)
    #     input = torch.unsqueeze(inputs[rand_index], 0).to(self.device)
    #     seq_id = indices[rand_index]
    #
    #     input, _ = val_model.image_encoder(input)
    #     info = self.datainfo[seq_id]
    #     image_name = info['name']
    #     truth_seq = [int(i) for i in info['seq'].split(" ")]
    #     truth_str = " ".join(self.decode_to_latex(truth_seq))
    #
    #     decoded_seq = greedy_search(val_model, input)
    #     bleu = compute_bleu(truth_seq, decoded_seq)
    #     decoded_str = " ".join(self.decode_to_latex(decoded_seq))
    #
    #     print('\n')
    #     self.std_logger.critical(
    #         f'Begin to validate the image {image_name} at epoch {epoch}')
    #     self.std_logger.info(truth_str)
    #     self.std_logger.info(decoded_str)
    #     print(
    #         f'length: encoded-{len(truth_seq)-2}, decoded-{len(decoded_seq)-2}, bleu score-{bleu}')

    def get_logger(self):
        return self.logger


def init_logger(log_dir, mode='w'):
    logger = logging.getLogger(log_dir)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    mode = 'a' if logger.handlers else 'w'

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    if log_dir != 'NoFile':
        file_handler = logging.FileHandler(log_dir, mode=mode)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(message)s', log_colors={
                                          'INFO': 'green', 'CRITICAL': 'red'})
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def compute_bleu(tf_tokens, decoded_tokens):
    bleu4_value = sentence_bleu([tf_tokens], decoded_tokens, weights=[
                                0.25, 0.25, 0.25, 0.25])
    return float(bleu4_value)


def get_decoded(decoded_tokens=[]):
    '''
    We have ensured that the first token is START_TOKEN
    We have also ensured that the last token is END or MAX_LEN
    '''
    if END_TOKEN not in decoded_tokens:
        return decoded_tokens
    else:
        return decoded_tokens[:decoded_tokens.index(END_TOKEN)+1]
