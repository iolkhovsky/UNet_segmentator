import cv2
import pickle

from voc_dataset.voc_index import VocIndex
from transform_utils import *


class VocSegmentationUNet:

    def __init__(self, path_to_binary_index, object_classes_list):
        assert type(object_classes_list) == list
        assert len(object_classes_list) > 0

        self.sample_ptr = 0
        self.object_classes = object_classes_list
        self.index = pickle.load(open(path_to_binary_index, 'rb'))
        self.input_tensor_shape = (3, 572, 572)
        self.output_tensor_shape = (len(self.object_classes) + 1, 388, 388)
        self.crop_y = int((self.input_tensor_shape[1] - self.output_tensor_shape[1]) / 2)
        self.crop_x = int((self.input_tensor_shape[2] - self.output_tensor_shape[2]) / 2)
        self.label2idx = {"background": 0}
        self.idx2label = {0: "background"}
        for idx, label in enumerate(self.object_classes):
            self.label2idx[label] = idx+1
            self.label2idx[idx+1] = label

        # for each example we should prepare output tensor + weights map (to avoid imballance of pixels cnt)
        loaded_img_paths = set()
        self.input_images = []
        self.target_tensors = []
        self.target_weights = []
        for target_class in object_classes_list:
            for sample_img_path, sample_map_path in self.index.index[target_class]:
                if sample_img_path not in loaded_img_paths:
                    loaded_img_paths.add(sample_img_path)
                    image = cv2.imread(sample_img_path)
                    image = cv2.resize(image, (self.input_tensor_shape[1], self.input_tensor_shape[2]))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = array_yxc2cyx(image)  # replace channels position - YXC to CYX
                    image = normalize_img_cyx(image)
                    raw_label_map = cv2.imread(sample_map_path)
                    raw_label_map = cv2.resize(raw_label_map, (self.input_tensor_shape[1], self.input_tensor_shape[2]))
                    crop_raw_label_map = raw_label_map[self.crop_y:self.crop_y+self.output_tensor_shape[1], \
                                         self.crop_x:self.crop_x+self.output_tensor_shape[2], :]
                    weights_map, classes_masks = self.get_masks(crop_raw_label_map, object_classes_list, \
                                                                self.index.color2label)
                    target_array = np.zeros(shape=self.output_tensor_shape, dtype=np.uint8)

                    target_array[0] = classes_masks["background"]
                    for class_label in object_classes_list:
                        class_id = self.label2idx[class_label]
                        target_array[class_id] = classes_masks[class_label]

                    self.input_images.append(image)
                    self.target_tensors.append(target_array)
                    self.target_weights.append(weights_map)
        self.input_images = np.asarray(self.input_images)
        self.target_tensors = np.asarray(self.target_tensors)
        self.target_weights = np.asarray(self.target_weights)
        return

    @staticmethod
    def get_masks(raw_label_map, classes, color2label):
        label2color = {}
        for color in color2label.keys():
            label2color[color2label[color]] = color
        weights_map = np.zeros(shape=(raw_label_map.shape[0], raw_label_map.shape[1]), dtype=np.float32)
        label_maps = {}
        pixels_cnt = {}
        classes_on_image = set()
        label_maps["background"] = np.zeros(shape=(raw_label_map.shape[0], raw_label_map.shape[1]), dtype=np.uint8)
        label_maps["background"] = label_maps["background"] + 1
        for label in classes:
            target_pixel_value = label2color[label]
            label_maps[label] = np.all(raw_label_map == np.asarray(target_pixel_value), axis=2).astype(np.uint8)
            label_maps["background"] = np.bitwise_and(label_maps["background"], \
                                                      (label_maps[label] == 0).astype(np.uint8))
            pixels_cnt[label] = np.sum(label_maps[label])
            if pixels_cnt[label]:
                classes_on_image.add(label)
        pixels_cnt["background"] = np.sum(label_maps["background"])

        background_presence = int(pixels_cnt["background"] > 0)
        average_weight = 1.0 / (len(classes_on_image) + background_presence)
        total_pixels = raw_label_map.shape[0] * raw_label_map.shape[1]

        if background_presence:
            weights_map = label_maps["background"] * average_weight / (pixels_cnt["background"] / total_pixels)
        for label in classes:
            if label in classes_on_image:
                weights_map += label_maps[label] * average_weight / (pixels_cnt[label] / total_pixels)
        return weights_map, label_maps

    def __len__(self):
        return len(self.input_images)

    def __make_sample(self, idx):
        return {"input": self.input_images[idx], "target": self.target_tensors[idx], "weight": self.target_weights}

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            return self.__make_sample(item)
        else:
            raise ValueError("Invalid index(-ices) to __get_item__ method")

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_ptr < len(self.input_images):
            out = self.sample_ptr
            self.sample_ptr += 1
            return self.__getitem__(out)
        else:
            self.sample_ptr = 0
            raise StopIteration

    def __str__(self):
        return "VOC-segmentation-dataset"

    def make_cv_frame_from_map(self, idx_map):
        out = np.zeros(shape=(self.output_tensor_shape.shape[1], self.output_tensor_shape.shape[2], 3))
        for idx in range(self.output_tensor_shape.shape[0]):
            if idx in self.id2label.keys():
                label = self.id2label[idx]
                color = self.index.label_map[label]
                out[idx_map == idx][0] = color[0]
                out[idx_map == idx][1] = color[1]
                out[idx_map == idx][2] = color[2]
        return out

    @staticmethod
    def decode_to_label_map(encoded_tensor):
        if type(encoded_tensor) == torch.tensor:
            encoded_tensor = encoded_tensor.detach().numpy()
        max_idx_map = np.argmax(encoded_tensor, axis=2)
        return max_idx_map


if __name__ == "__main__":
    voc_dataset_index_path = "/home/igor/github/my/UNet_segmentator/voc_dataset/voc_segmentation_index.dat"
    dataset = VocSegmentationUNet(voc_dataset_index_path, ["person"])