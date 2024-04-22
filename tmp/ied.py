import numpy as np
from Levenshtein import *
from warnings import warn
import distance as dis


class StringMatcher:
    """A SequenceMatcher-like class built on the top of Levenshtein"""

    def _reset_cache(self):
        self._ratio = self._distance = None
        self._opcodes = self._editops = self._matching_blocks = None

    def __init__(self, isjunk=None, seq1='', seq2=''):
        if isjunk:
            warn("isjunk not NOT implemented, it will be ignored")
        self._str1, self._str2 = seq1, seq2
        self._reset_cache()

    def set_seqs(self, seq1, seq2):
        self._str1, self._str2 = seq1, seq2
        self._reset_cache()

    def set_seq1(self, seq1):
        self._str1 = seq1
        self._reset_cache()

    def set_seq2(self, seq2): 
        self._str2 = seq2
        self._reset_cache()

    def get_opcodes(self):
        if not self._opcodes:
            if self._editops:
                self._opcodes = opcodes(self._editops, self._str1, self._str2)
            else:
                self._opcodes = opcodes(self._str1, self._str2)
        return self._opcodes

    def get_editops(self):
        if not self._editops:
            if self._opcodes:
                self._editops = editops(self._opcodes, self._str1, self._str2)
            else:
                self._editops = editops(self._str1, self._str2)
        return self._editops

    def get_matching_blocks(self):
        if not self._matching_blocks:
            self._matching_blocks = matching_blocks(self.get_opcodes(),
                                                    self._str1, self._str2)
        return self._matching_blocks

    def ratio(self):
        if not self._ratio:
            self._ratio = ratio(self._str1, self._str2)
        return self._ratio

    def quick_ratio(self):
        # This is usually quick enough :o)
        if not self._ratio:
            self._ratio = ratio(self._str1, self._str2)
        return self._ratio

    def real_quick_ratio(self):
        len1, len2 = len(self._str1), len(self._str2)
        return 2.0 * min(len1, len2) / (len1 + len2)

    def distance(self):
        if not self._distance:
            self._distance = distance(self._str1, self._str2)
        return self._distance


def img_edit_distance(im1, im2):
    img_data1 = np.asarray(im1, dtype=np.uint8)  # height, width
    img_data1 = np.transpose(img_data1)
    h1 = img_data1.shape[1]
    w1 = img_data1.shape[0]
    img_data1 = (img_data1 <= 128).astype(np.uint8)

    img_data2 = np.asarray(im2, dtype=np.uint8)  # height, width
    img_data2 = np.transpose(img_data2)
    h2 = img_data2.shape[1]
    w2 = img_data2.shape[0]
    img_data2 = (img_data2 <= 128).astype(np.uint8)

    if h1 == h2:
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]
    elif h1 > h2:  # pad h2
        seq1 = [''.join([str(i) for i in item]) for item in img_data1]
        seq2 = [''.join([str(i) for i in item])+''.join(['0']*(h1-h2))
                for item in img_data2]
    else:
        seq1 = [''.join([str(i) for i in item])+''.join(['0']*(h2-h1))
                for item in img_data1]
        seq2 = [''.join([str(i) for i in item]) for item in img_data2]

    seq1_int = [int(item, 2) for item in seq1]
    seq2_int = [int(item, 2) for item in seq2]
    big = int(''.join(['0' for i in range(max(h1, h2))]), 2)
    seq1_eliminate = []
    seq2_eliminate = []
    seq1_new = []
    seq2_new = []
    for idx, items in enumerate(seq1_int):
        if items > big:
            seq1_eliminate.append(items)
            seq1_new.append(seq1[idx])
    for idx, items in enumerate(seq2_int):
        if items > big:
            seq2_eliminate.append(items)
            seq2_new.append(seq2[idx])
    if len(seq2) == 0:
        return (len(seq1), len(seq1), False, False)

    def make_strs(int_ls, int_ls2):
        d = {}
        seen = []

        def build(ls):
            for l in ls:
                if int(l, 2) in d:
                    continue
                found = False
                l_arr = np.array([int(i) for i in l])

                for l2, l2_arr in seen:
                    if np.abs(l_arr - l2_arr).sum() < 5:
                        d[int(l, 2)] = d[int(l2, 2)]
                        found = True
                        break
                if not found:
                    d[int(l, 2)] = chr(len(seen))
                    seen.append((l, np.array([int(i) for i in l])))

        build(int_ls)
        build(int_ls2)
        return "".join([d[int(l, 2)] for l in int_ls]), "".join([d[int(l, 2)] for l in int_ls2])
    # if out_path:
    seq1_t, seq2_t = make_strs(seq1, seq2)

    edit_distance = dis.levenshtein(seq1_int, seq2_int)
    match = True
    if edit_distance > 0:
        matcher = StringMatcher(None, seq1_t, seq2_t)

        ls = []
        for op in matcher.get_opcodes():
            if op[0] == "equal" or (op[2]-op[1] < 5):
                ls += [[int(r) for r in l]
                       for l in seq1[op[1]:op[2]]
                       ]
            elif op[0] == "replace":
                a = seq1[op[1]:op[2]]
                b = seq2[op[3]:op[4]]
                ls += [[int(r1)*3 + int(r2)*2
                        if int(r1) != int(r2) else int(r1)
                        for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                          b[i] if i < len(b) else [0]*1000)]
                       for i in range(max(len(a), len(b)))]
                match = False
            elif op[0] == "insert":

                ls += [[int(r)*3 for r in l]
                       for l in seq2[op[3]:op[4]]]
                match = False
            elif op[0] == "delete":
                match = False
                ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

    match1 = match
    seq1_t, seq2_t = make_strs(seq1_new, seq2_new)

    if len(seq2_new) == 0 or len(seq1_new) == 0:
        if len(seq2_new) == len(seq1_new):
            # all blank
            return (edit_distance, max(len(seq1_int), len(seq2_int)), match1, True)
        return (edit_distance, max(len(seq1_int), len(seq2_int)), match1, False)
    match = True
    matcher = StringMatcher(None, seq1_t, seq2_t)

    ls = []
    for op in matcher.get_opcodes():
        if op[0] == "equal" or (op[2]-op[1] < 5):
            ls += [[int(r) for r in l]
                   for l in seq1[op[1]:op[2]]
                   ]
        elif op[0] == "replace":
            a = seq1[op[1]:op[2]]
            b = seq2[op[3]:op[4]]
            ls += [[int(r1)*3 + int(r2)*2
                    if int(r1) != int(r2) else int(r1)
                    for r1, r2 in zip(a[i] if i < len(a) else [0]*1000,
                                      b[i] if i < len(b) else [0]*1000)]
                   for i in range(max(len(a), len(b)))]
            match = False
        elif op[0] == "insert":

            ls += [[int(r)*3 for r in l]
                   for l in seq2[op[3]:op[4]]]
            match = False
        elif op[0] == "delete":
            match = False
            ls += [[int(r)*2 for r in l] for l in seq1[op[1]:op[2]]]

    match2 = match

    return (edit_distance, max(len(seq1_int), len(seq2_int)), match1, match2)
