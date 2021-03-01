import os
import sys
from os import listdir
from os.path import isfile, join
from imghdr import what
import multiprocessing as mp
from PIL import Image
from functools import partial
from tqdm import tqdm
from shutil import copy
from imagehash import dhash


class Filter:
    def __init__(self, work_place=os.getcwd(), n_proc=1, keep_original=True):
        self.work_place = work_place
        self.n_proc = n_proc
        self.keep_original = keep_original

    def chunks(self, src_list):
        n = len(src_list) // self.n_proc + 1
        if n > 500:
            n = 500
        for i in range(0, len(src_list), n):
            yield src_list[i:i + n]

    @staticmethod
    def copy(src, dst):
        if not os.path.exists(dst):
            copy(src, dst)

    def mkdir(self, target_sub_path, out_sub_path):
        target_path = os.path.join(self.work_place, target_sub_path)
        out_sub_path = os.path.join(self.work_place, out_sub_path)
        if not os.path.exists(out_sub_path):
            os.mkdir(out_sub_path)
        file_list = [os.path.join(target_path, f) for f in listdir(target_path) if isfile(join(target_path, f))]
        print("[*] There are {} image. Going to run with {} threads.".format(len(file_list), self.n_proc))
        return target_sub_path, out_sub_path, file_list

    def hash_filter_handler(self, hash_set, lock, out_sub_path, files_chunk):
        for file_path in tqdm(files_chunk, desc="Filtering using hash", unit="file"):
            f = open(file_path, 'rb')
            content = f.read()
            hash_res = hash(content)
            f.close()
            lock.acquire()
            if hash_res not in hash_set:
                hash_set.add(hash_res)
                lock.release()
                file_name = os.path.basename(file_path)
                self.copy(file_path, os.path.join(out_sub_path, file_name))
            else:
                lock.release()

    def hash_filter(self, target_sub_path, out_sub_path):
        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)
        unique_hash_set = set()
        lock = mp.Manager().Lock()
        with mp.Pool(processes=self.n_proc) as pool:
            pool.map(partial(self.hash_filter_handler, unique_hash_set, lock, out_sub_path), self.chunks(file_list))
        print("[*] Files with unique hash have been copied to：{}".format(out_sub_path))

    def valid_handler(self, out_sub_path, files_chunk):
        valid_types = {"jpeg", "jpg", "png"}
        for file_path in tqdm(files_chunk, desc="Filtering using file type", unit="file"):
            file_type = what(file_path)
            if file_type in valid_types:
                base_name = os.path.basename(file_path)
                file_name = os.path.splitext(base_name)
                file_name = file_name[0] + "." + file_type
                self.copy(file_path, os.path.join(out_sub_path, file_name))

    def valid_filter(self, target_sub_path, out_sub_path):

        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)

        with mp.Pool(processes=self.n_proc) as pool:
            pool.map(partial(self.valid_handler, out_sub_path), self.chunks(file_list))
        print("[*] Images have been copied to: {}".format(out_sub_path))

    def size_handler(self, out_sub_path, threshold, files_chunk):
        for file_path in tqdm(
                files_chunk,
                desc="Filtering files with size less than {}KB".format(threshold / 1024),
                unit="files"):
            if os.path.getsize(file_path) > threshold:
                file_name = os.path.basename(file_path)
                self.copy(file_path, os.path.join(out_sub_path, file_name))

    def size_filter(self, target_sub_path, out_sub_path, threshold=10 * 1024):

        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)

        with mp.Pool(processes=self.n_proc) as pool:
            pool.map(partial(self.size_handler, out_sub_path, threshold), self.chunks(file_list))
        print("[*] Images greater than {}KB have been copied to: {}".format(threshold / 1024, out_sub_path))

    def diff_hash_filter_handler(self, hash_set, lock, out_sub_path, files_chunk):
        for file_path in tqdm(
                files_chunk,
                desc="Filtering using dhash",
                unit="image"):
            image = Image.open(file_path)
            d_hash = dhash(image)
            hash_res = d_hash.__hash__()
            lock.acquire()
            print(hash_set)
            if hash_res not in hash_set:
                hash_set.add(hash_res)
                lock.release()
                file_name = os.path.basename(file_path)
                self.copy(file_path, os.path.join(out_sub_path, file_name))
            else:
                lock.release()

    def _filter_handler(self, lock, hash_res, hash_set, file_path, out_sub_path):
        lock.acquire()
        print(hash_set)
        if hash_res not in hash_set:
            hash_set.add(hash_res)
            lock.release()
            file_name = os.path.basename(file_path)
            self.copy(file_path, os.path.join(out_sub_path, file_name))
        else:
            lock.release()

    def diff_hash_filter(self, target_sub_path, out_sub_path):
        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)
        unique_hash_set = set()
        lock = mp.Manager().Lock()
        with mp.Pool(processes=1) as pool:
            pool.map(partial(self.diff_hash_filter_handler, unique_hash_set, lock, out_sub_path),
                     self.chunks(file_list))
        print("[*] Images with unique dhash have been copied to: {}".format(out_sub_path))

    @staticmethod
    def hamming_d(bin_a, bin_b):
        """Calculate the Hamming distance between two bit strings"""
        count, z = 0, bin_a ^ bin_b
        while z:
            count += 1
            z &= z - 1  # magic!X
        return count

    def hamming_filter(self, target_sub_path, out_sub_path, threshold=0.1):
        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)
        keep_list = set()
        image = Image.open(file_list[0])
        keep_list.add(dhash(image).__hash__())
        self.copy(file_list[0], os.path.join(out_sub_path, os.path.basename(file_list[0])))

        for file_path in tqdm(file_list, desc="Filtering using Hamming Distance", unit="image"):
            image = Image.open(file_path)
            d_hash = dhash(image).__hash__()
            if d_hash is None:
                continue
            should_save = True

            for now_have in keep_list:
                diff = self.hamming_d(now_have, d_hash)
                diff_ratio = diff / 64
                if diff_ratio < threshold:
                    should_save = False
                    break
            if should_save:
                file_name = os.path.basename(file_path)
                keep_list.add(d_hash)
                self.copy(file_path, os.path.join(out_sub_path, file_name))
        print("[*] Images filtered by Hamming Distance have been copied to: {}".format(out_sub_path))


if __name__ == '__main__':
    new_filter = Filter(
        work_place=sys.argv[1],
        n_proc=4,
        keep_original=True)
    new_filter.size_filter("./", "big_enough", threshold=10 * 1024)
    new_filter.valid_filter("big_enough", "valid_picture")
    new_filter.hash_filter("valid_picture", "unique_hash")
    new_filter.diff_hash_filter("unique_hash", "unique_dhash")
    new_filter.hamming_filter("unique_dhash", "hamming", 0.1)
    # new_filter.valid_filter("big_enough", "valid_picture")
