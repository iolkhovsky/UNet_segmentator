from voc_dataset.voc_index import VocIndex
import pickle


class VocSegmentationUNet:

    def __init__(self, path_to_binary_index, object_classes_list):
        self.object_classes = object_classes_list
        self.sample_ptr = 0
        self.index = pickle.load(open(path_to_binary_index, 'rb'))
        self.input_tensor_shape = (3, 572, 572)
        self.output_tensor_shape = (2, 388, 388)
        return

    def __len__(self):
        return len(self.index)

    def __make_sample(self, idx):
        img_path, mask_path, classes = self.index[idx]
        sample_desc = self.index[idx]
        path = sample_desc.abs_path
        image = cv2.imread(path)
        image = cv2.resize(image, (self.target_size[0], self.target_size[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # reverse bgr to rgb
        image = array_yxc2cyx(image)  # replace channels position - YXC to CYX
        image = normalize_img_cyx(image)
        in_tensor = torch.from_numpy(image)

        # head tensors
        cached_obj = self.cache.try_get(idx)
        if cached_obj:
            return {"input": in_tensor, "target": cached_obj}

        target_tensors = self.codec.encode(sample_desc)
        self.cache.add(idx, target_tensors)
        return {"input": in_tensor, "target": target_tensors}

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
        if self.sample_ptr < len(self.index):
            out = self.sample_ptr
            self.sample_ptr += 1
            return self.__getitem__(out)
        else:
            self.sample_ptr = 0
            raise StopIteration

    def __str__(self):
        return "VOC-segmentation-dataset"

