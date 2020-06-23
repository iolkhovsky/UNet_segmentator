from glob import glob
from os.path import join, isdir
import cv2
import datetime
import pickle
import sys

from voc_dataset.utils import load_label_map, make_color2label, color_histogram, make_id2label


class VocIndex:

    def __init__(self, root, label_map_path):
        self.label_map = load_label_map(label_map_path)
        self.color2label = make_color2label(self.label_map)
        self.id2label = make_id2label(self.label_map)
        self.root = root
        self.index = {}
        self.linear_index = []
        self.idx = 0
        if not isdir(self.root):
            raise ValueError("Invalid dataset path")
        self.all_images_list = glob(join(self.root, "JPEGImages/*.jpg"))
        self.all_segmentation_maps_list = glob(join(self.root, "SegmentationClass/*.png"))
        with open(join(self.root, "ImageSets/Segmentation/train.txt"), "r") as f:
            lines = f.readlines()
            segmented_images = [x.strip() for x in lines]
        for sample_id, sample_name in enumerate(segmented_images):
            sample_img_path = self.root + "/JPEGImages/" + sample_name + ".jpg"
            sample_map_path = self.root + "/SegmentationClass/" + sample_name + ".png"
            sample_map = cv2.imread(sample_map_path)
            hist = color_histogram(sample_map)
            objects = []
            for hist_val in hist.keys():
                if hist_val in self.color2label.keys():
                    label = self.color2label[hist_val]
                    if label not in self.index.keys():
                        self.index[label] = []
                    objects.append(label)
                    self.index[label].append((sample_img_path, sample_map_path))
            self.linear_index.append((sample_img_path, sample_map_path, objects))
            for i in range(1, 11):
                if sample_id == int(len(segmented_images) * i * 0.1) - 1:
                    self.log(str(i*10), "% loaded")
        return

    def __getitem__(self, item):
        return self.linear_index[item]

    def __str__(self):
        return "Pascal VOC for segmentation"

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self.linear_index):
            out = self.linear_index[self.idx]
            self.idx += 1
            return out
        else:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return len(self.linear_index)

    @staticmethod
    def log(*args):
        st = str(datetime.datetime.now())
        st += "\t<Index-VOC-Segmentation>: "
        for t in args:
            st += t + " "
        print(st)


if __name__ == "__main__":
    print(sys.argv)
    index = VocIndex("/home/igor/datasets/VOC_2012/trainval",
                     "/home/igor/github/my/UNet_segmentator/configs/label_map.txt")
    index_target_path = "voc_segm_index.dat"
    pickle.dump(index, open(index_target_path, 'wb'))
    index.log("Index has been saved at " + index_target_path)
