import urllib.request
import os
import random

WORD_URL = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
DUMP_PATH = os.path.join(os.path.curdir, 'words.csv')


def generate_human_readable_id():
    words = download_words() if not os.path.exists(DUMP_PATH) else load_words()
    id = '-'.join([random.choice(words) for _ in range(3)])
    return id


def load_words():
    words = []
    with open(DUMP_PATH) as f:
        words.append(f.readline().strip())
    return words


def download_words():
    response = urllib.request.urlopen(WORD_URL)
    long_txt = response.read().decode()
    words = long_txt.splitlines()
    with open(DUMP_PATH, 'w') as f:
        [f.write(word + '\n') for word in words]
    return words
