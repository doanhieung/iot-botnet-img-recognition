import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed

N_JOBS = cpu_count()
IMAGE_SIZE = (64, 64)


def get_size(size):
    # source Malware images: visualization and automatic classification by L. Nataraj
    # url : http://dl.acm.org/citation.cfm?id=2016908
    if (size < 10240):
        width = 32
    elif (10240 <= size <= 10240 * 3):
        width = 64
    elif (10240 * 3 <= size <= 10240 * 6):
        width = 128
    elif (10240 * 6 <= size <= 10240 * 10):
        width = 256
    elif (10240 * 10 <= size <= 10240 * 20):
        width = 384
    elif (10240 * 20 <= size <= 10240 * 50):
        width = 512
    elif (10240 * 50 <= size <= 10240 * 100):
        width = 768
    else:
        width = 1024
    height = int(size / width) + 1
    return (width, height)


def extract_image(path):
    file_name = path.split('/')[-1]
    output = args.output + file_name + '.png'
    bin = []
    with open(path, 'rb') as f:
        data = f.read(1)
        while data != b'':
            bin.append(ord(data))
            data = f.read(1)

    size = get_size(len(bin))
    image = Image.new('L', size)
    image.putdata(bin)
    image = image.resize(IMAGE_SIZE)
    image.save(output)


parser = argparse.ArgumentParser(description="Extract binary images.")
parser.add_argument("--input", "-i",
                    nargs="?",
                    help="Input folder.")

parser.add_argument("--output", "-o",
                    nargs="?",
                    help="Output folder.")
args = parser.parse_args()

inputs = glob(args.input + '*')
Parallel(n_jobs=N_JOBS)(
    delayed(extract_image)(path) for path in tqdm(inputs))
