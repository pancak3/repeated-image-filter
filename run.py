import os
from os import listdir
from os.path import isfile, join
from imghdr import what
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from shutil import move, copy
from imagededup.methods import DHash


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
        print("[*] 共有 {} 个文件，使用 {} 线程处理。".format(len(file_list), self.n_proc))
        return target_sub_path, out_sub_path, file_list

    def hash_filter_handler(self, hash_set, lock, out_sub_path, files_chunk):
        for file_path in tqdm(files_chunk, desc="正在检验是否拥有唯一哈希", unit=" 文件"):
            f = open(file_path, 'rb')
            content = f.read()
            f_hash = hash(content)
            f.close()

            lock.acquire()
            if f_hash not in hash_set:
                hash_set.add(f_hash)
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
        print("[*] 唯一哈希文件皆已存于：{}".format(out_sub_path))

    def valid_handler(self, out_sub_path, files_chunk):
        valid_types = {"jpeg", "jpg", "png"}
        for file_path in tqdm(files_chunk, desc="正在检验是否为图像文件", unit=" 文件"):
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
        print("[*] 有效图像文件皆已存于：{}".format(out_sub_path))

    def size_handler(self, out_sub_path, threshold, files_chunk):
        for file_path in tqdm(files_chunk, desc="正在检验是否大于 {}KB".format(threshold / 1024), unit=" 文件"):
            if os.path.getsize(file_path) > threshold:
                file_name = os.path.basename(file_path)
                self.copy(file_path, os.path.join(out_sub_path, file_name))

    def size_filter(self, target_sub_path, out_sub_path, threshold=10 * 1024):

        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)

        with mp.Pool(processes=self.n_proc) as pool:
            pool.map(partial(self.size_handler, out_sub_path, threshold), self.chunks(file_list))
        print("[*] 大于 {}KB 图像文件皆已存于：{}".format(threshold / 1024, out_sub_path))

    def diff_hash_filter_handler(self, hash_set, lock, out_sub_path, files_chunk):
        d_hash_method = DHash()
        for file_path in tqdm(files_chunk, desc="正在检验是否拥有差异哈希", unit=" 图像"):
            d_hash = d_hash_method.encode_image(image_file=file_path)

            lock.acquire()
            if d_hash not in hash_set:
                hash_set.add(d_hash)
                lock.release()
                file_name = os.path.basename(file_path)
                self.copy(file_path, os.path.join(out_sub_path, file_name))
            else:
                lock.release()

    def diff_hash_filter(self, target_sub_path, out_sub_path):

        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)

        unique_hash_set = set()
        lock = mp.Manager().Lock()
        with mp.Pool(processes=self.n_proc) as pool:
            pool.map(partial(self.diff_hash_filter_handler, unique_hash_set, lock, out_sub_path),
                     self.chunks(file_list))
        print("[*] 唯一差异哈希图像皆已存于：{}".format(out_sub_path))

    @staticmethod
    def hamming_d(bin_a, bin_b):
        """Calculate the Hamming distance between two bit strings"""
        assert len(bin_a) == len(bin_b)
        count, z = 0, int(bin_a, 2) ^ int(bin_b, 2)
        while z:
            count += 1
            z &= z - 1  # magic!X
        return count

    def hamming_filter(self, target_sub_path, out_sub_path, threshold=0.1):

        target_sub_path, out_sub_path, file_list = self.mkdir(target_sub_path, out_sub_path)
        d_hash_method = DHash()
        keep_list = set()
        keep_list.add(d_hash_method.encode_image(image_file=file_list[0]))
        self.copy(file_list[0], os.path.join(out_sub_path, os.path.basename(file_list[0])))

        for file_path in tqdm(file_list, desc="差异哈希汉明距离", unit="图像"):
            d_hash = d_hash_method.encode_image(image_file=file_path)
            if d_hash is None:
                continue
            d_hash_bin = bin(int(d_hash, 16))[2:].zfill(64)
            should_save = True

            for now_have in keep_list:
                now_have = bin(int(now_have, 16))[2:].zfill(64)
                diff = self.hamming_d(now_have, d_hash_bin)
                diff_ratio = diff / 64
                if diff_ratio < threshold:
                    should_save = False
                    break
            if should_save:
                file_name = os.path.basename(file_path)
                keep_list.add(d_hash)
                self.copy(file_path, os.path.join(out_sub_path, file_name))
        print("[*] 经汉明过滤图像皆已存于：{}".format(out_sub_path))


if __name__ == '__main__':
    new_filter = Filter(work_place="../../dataset/images/filtered", n_proc=4, keep_original=True)
    # new_filter.size_filter("unique_dhash", "big_enough", threshold=10 * 1024)
    # new_filter.valid_filter("big_enough", "valid_picture")
    # new_filter.hash_filter("valid_picture", "unique_hash")
    # new_filter.diff_hash_filter("unique_hash", "unique_dhash")
    new_filter.hamming_filter("unique_dhash", "hamming", 0.1)
