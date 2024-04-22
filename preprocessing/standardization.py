REPLACE = {
    322: 290,  # \operatorname* to \mathrm
    165: 286,  # \cal to \mathcal
    # 141: 284,  # \bf to \mathbf
    270: 254,  # \leq to \le
    433: 372,  # \to to \rightarrow
    129: 9,  # \ast to *
    455: 506,  # \vert to |
    294: 506,  # \mid to |
    117: 469,
    400: 472,  # \sp to ^
    378: 473,  # \sb to _
    # left
    255: 7,  # \left( to (
    258: 69,  # \left[ to [
    264: 468,  # \left\{ to \{
    263: 506,  # \left\vert to |
    269: 506,  # \left| to |
    265: 469,  # \left\| to \|

    # right
    361: 8,  # \right) to )
    371: 471,  # \right] to ]
    370: 470,  # \right\} to \}
    368: 506,  # \right\vert to |
    375: 506,  # \right| to |
    369: 469,  # \right\| to \|
}


COMPLEX_BLANK1 = [
    230,  # \hspace {}
    282,  # \makebox {}
]
COMPLEX_BLANK2 = [
    243,  # \kern {}
    296,  # \mkern {}
]

BLANKS = [
    71,  # \
    72,  # \!
    77,  # \,
    80,  # \:
    81,  # \;
    348,  # \quad
    349,  # \qquad
    401,  # \space
    508,  # ~
]

SPATIAL_LIST = [
    472,  # ^
    473,  # _

    215,  # \frac
    405,  # \sqrt
    321,  # \operatorname
    284,  # \mathbf
    286,  # \mathcal
    290,  # \mathrm
    134,  # \bar
    223,  # \hbar
    222,  # \hat
    463,  # \widehat
    430,  # \tilde
    464,  # \widetilde
    170,  # \check

    285,  # \mathbin
    287,  # \mathit
    288,  # \mathop
    289,  # \mathrel
    291,  # \mathsf
]


def replace_char(my_string=[]):
    result = []
    for i in my_string:
        if i in REPLACE:
            result.append(REPLACE[i])
        else:
            result.append(i)
    return result


def find_matching_brackets(my_string=[]):
    stack = []
    result = {}

    for i in range(len(my_string)):
        word = my_string[i]
        # We meet {
        if word == 505:
            stack.append(i)
        # We meet }
        elif word == 507:
            if stack:
                open_bracket_position = stack.pop()
                result[open_bracket_position] = i

    return result, {v:k for k, v in result.items()}

def filter_blank(my_string=[]):
    brackets, _ = find_matching_brackets(my_string)
    length = len(my_string)
    enable = [True for _ in my_string]
    for i in range(length):
        index = my_string[i]
        if not enable[i]:
            continue
        if index in BLANKS: enable[i] = False
        # remove the complex blank
        elif index in COMPLEX_BLANK1:
            if i+1 < length:
                if my_string[i+1] == 505 and (i+1) in brackets:
                    for j in range(i, brackets[i+1]+1):
                        enable[j] = False

        elif index in COMPLEX_BLANK2:
            # i-1 have to > length
            if my_string[i-1] == 505 and (i-1) in brackets:
                for j in range(i-1, brackets[i-1]+1):
                    enable[j] = False

    result = []
    for i in range(length):
        index = my_string[i]
        if enable[i]:
            result.append(index)
    return result


def clean_seq(my_string=[], filter_blank=False):
    my_string = replace_char(my_string)
    brackets, inverted_brackets = find_matching_brackets(my_string)
    length = len(my_string)
    result = []
    enable = [True for _ in my_string]
    for i in range(length):
        index = my_string[i]
        if not enable[i]:
            continue

        # blank filtering
        if filter_blank:
            if index in BLANKS:
                enable[i] = False
            # remove the complex blank
            elif index in COMPLEX_BLANK1:
                if i+1 < length:
                    if my_string[i+1] == 505 and (i+1) in brackets:
                        for j in range(i, brackets[i+1]+1):
                            enable[j] = False

            elif index in COMPLEX_BLANK2:
                # i-1 have to > length
                if my_string[i-1] == 505 and (i-1) in brackets:
                    for j in range(i-1, brackets[i-1]+1):
                        enable[j] = False

        # can not be the first because the seq start with [SOS]
        # abundant {x}, let's find the matching '{}' and remove them
        if my_string[i-1] == 505 and index != 141:
            # remove {}
            if my_string[i] == 507:
                enable[i-1] = False
                enable[i] = False

            # frac{x}{y}, do not remove the blank of y
            if my_string[i-2] == 507 and (i-2) in inverted_brackets:
                    left = inverted_brackets[i-2] 
                    # if not frac structure
                    if my_string[left-1] != 215:
                        if (i-1) in brackets:
                            enable[i-1] = False
                            enable[brackets[i-1]] = False
                        # do nothing, this blank cannot be removed
            # i-2 can not be blank
            elif my_string[i-2] not in SPATIAL_LIST:
                # if we have a blank match for this {
                if (i-1) in brackets:
                    enable[i-1] = False
                    enable[brackets[i-1]] = False


    # formatting the \x{}
    i = 0
    while i < length:
        index = my_string[i]
        if enable[i]:
            result.append(index)

        if index in SPATIAL_LIST:
            # check the follower
            if i+1 < length:
                next_index = my_string[i+1]
                if next_index != 505:
                    result.append(505)
                    result.append(next_index)
                    result.append(507)
                    i += 1
                    # if it is \frac we have to check one more
                    if i+1 < length and index == 215:
                        next_next_index = my_string[i+1]
                        if next_next_index != 505:
                            result.append(505)
                            result.append(next_next_index)
                            result.append(507)
                            i += 1
        i += 1
    return result


def print_list(vocab=[]):
    for k, v in REPLACE.items():
        print(f'{vocab[k]}: {vocab[v]}')
    print([vocab[i] for i in BLANKS])
    print([vocab[i] for i in SPATIAL_LIST])
    print([vocab[i] for i in COMPLEX_BLANK1])
    print([vocab[i] for i in COMPLEX_BLANK2])
