import numpy as np

# Transposition Ciphers
def place(text, key, i, grid):
    # Used for railfence
    boundary = key - 1
    lim = boundary * 2
    split = 1 if (i % lim) < boundary else 2

    if split == 1:
        pos = i % boundary
        grid[pos][i] = text[i]
    elif split == 2:
        pos = boundary - (i % boundary)
        grid[pos][i] = text[i]

def railfence(text, key):
    grid = [['' for _ in range(len(text))] for _ in range(key)]

    for i in range(len(text)):
        place(text, key, i, grid)
    
    word = []
    for i in range(key):
        for j in range(len(text)):
            if grid[i][j] != '':
                word.append(grid[i][j])

    return ''.join(word)

def irreg_columnar(text, key):
    row_len = len(key)
    col_len = (len(text) // row_len) + 1
    grid = np.empty((col_len, row_len), dtype = str)
    chars = np.array([c for c in key])
    order = np.argsort(chars)

    count = 0
    for i in range(col_len):
        for j in range(row_len):
            if count == len(text):
                break

            grid[i][j] = text[count]

            count += 1

    grid = grid[:, order]
    word = [char for row in grid for char in row]
    return ''.join(word)

# Substitution Ciphers
def caesar(text, key):
    word = np.array([c for c in text])

    ord_vec = np.vectorize(ord)
    chr_vec = np.vectorize(chr)

    word = (ord_vec(word) + key) % 123 
    word[word < 97] += 97
    word = chr_vec(word).tolist()

    return ''.join(word)

# Polyalphabetic Ciphers
def beaufort(text, key):
    t = len(text)
    k = len(key)

    if t <= k:
        overlap = key[:t]
    else:
        overlap = [c for c in key]
        keys_left = t - k
        count = 0
        while count < keys_left:
            for i in key:
                if count >= keys_left:
                    break
                overlap.append(i)
                count += 1
        overlap = ''.join(overlap)

    ref = [chr(97 +  i) for i in range(26)]

    word = []
    for i in range(t):
        if text[i] < overlap[i]:
            dist = ord(overlap[i]) - ord(text[i])
        elif text[i] == overlap[i]:
            dist = 0
        else:
            dist = ord(text[i]) - ord(overlap[i])
            dist = 26 - dist

        word.append(ref[dist])
    
    return ''.join(word)

def autokey(text, key):
    t = len(text)
    k = len(key)

    if t <= k:
        overlap = key[:t]
    else:
        overlap = [c for c in key]
        keys_left = t - k
        count = 0
        while count < keys_left:
            for i in text:
                if count >= keys_left:
                    break
                overlap.append(i)
                count += 1
        overlap = ''.join(overlap)

    ref = np.array([chr(97 + i) for i in range(26)])

    word = []
    for i in range(t):
        offset = np.argsort(ref == overlap[i])[-1]
        comp = ord(text[i]) + offset
        if comp >= 123:
            char = chr((comp % 123) + 97)
        else:
            char = chr(comp)
        word.append(char)

    return ''.join(word)

# Polygraphic Substitution Ciphers
def hill(text, key):
    np.random.seed(key)
    size = np.random.randint(2, 4)
    matrix = np.random.randint(5, 100, (size, size))

    size = matrix.shape[0]
    tokenize = np.vectorize(ord)
    chars = np.array([c for c in text])
    tokens = tokenize(chars) - 97

    num_vec = len(text) // size
    if len(text) % size:
        num_vec += 1

    pad = size - (len(text) % size)
    tokens = list(tokens)
    for _ in range(pad):
        tokens.append(2)
    tokens = np.array(tokens)

    detokenize = np.vectorize(chr)
    word = []
    for i in range(num_vec):
        vec = tokens[i * size : (i + 1) * size]
        comp = (np.dot(matrix, vec) % 26) + 97
        letters = list(detokenize(comp))
        word.append(letters)

    word = [char for row in word for char in row]

    return ''.join(word)

# Polyciphers
def columnar_autokey(text, key):
    return irreg_columnar(autokey(text, key[0]), key[1])

def rail_hill(text, key):
    return railfence(hill(text, key[0]), key[1])

def beaufort_hill(text, key):
    return beaufort(hill(text, key[0]), key[1])

def hill_autokey(text, key):
    return hill(autokey(text, key[0]), key[1])

def beaufort_rail(text, key):
    return beaufort(railfence(text, key[0]), key[1])

# Paper Encryption
def special_caesar(t1, t2, key):
    # Used for paper encryption
    odd = np.array([c for c in t1])
    even = np.array([c for c in t2])

    ord_vec = np.vectorize(ord)
    chr_vec = np.vectorize(chr)

    offsets = np.arange(key, 0, -1)
    odd = ord_vec(odd)
    even = ord_vec(even)
    for i in range(len(odd)):
        odd[-(i + 1)] += offsets[i]
    for i in range(len(even)):
        even[i] += offsets[i]

    odd %= 123
    even %= 123

    odd[odd < 97] += 97
    even[even < 97] += 97

    odd_word, even_word = [], []
    odd_word = chr_vec(odd).tolist()
    even_word = chr_vec(even).tolist()

    return ''.join(odd_word), ''.join(even_word)

def pos_replace(text):
    # Used for paper encryption
    word = np.array([c for c in text])
    length = len(text)
    first = 0
    middle = (length // 2) - 1
    last = length - 1
    idx = np.array([first, middle, last])

    ord_vec = np.vectorize(ord)
    chr_vec = np.vectorize(chr)

    word = ord_vec(word)
    word[idx] += length - 26

    ref = np.arange(97, 123, 1)
    vals = word[word < 97]
    distances = 97 - vals
    word[word < 97] = ref[-distances]
    word = chr_vec(word)

    return ''.join(word.tolist())

def advanced_sub(text, key):
    # Final Encryption
    if len(text) < 3:
        return text
    keylength = len(text)
    new_text = np.array([c for c in text])

    even_pos = np.arange(0, keylength, 2)
    odd_pos = np.arange(1, keylength, 2)

    evenGroup = new_text[even_pos]
    oddGroup = new_text[odd_pos]

    evenGroup = ''.join(list(evenGroup)[::-1])
    oddGroup = ''.join(list(oddGroup)[::-1])

    groups = special_caesar(evenGroup, oddGroup, keylength)
    word = ''.join(groups)

    word = pos_replace(word)

    word = hill(text, 100)

    return word