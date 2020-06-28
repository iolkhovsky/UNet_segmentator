from os.path import isfile


def load_label_map(path_to_config):
    if not isfile(path_to_config):
        raise FileNotFoundError("Invalid path to label map file")
    out = {}
    with open(path_to_config, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split()
            if len(words) != 5:
                error_msg = "Label map has invalid structure. " + \
                    "Target: id:int, label:string, b:int, g:int, r:int" + \
                    " split symbol is space. + Got "+str(len(words))+" words"
                raise RuntimeError(error_msg)
            idx, label = int(words[0]), words[1]
            b, g, r = int(words[2]), int(words[3]), int(words[4])
            out[label] = {"id": idx, "color": (b, g, r)}
    return out


def make_id2label(label_map):
    out = {}
    for label in label_map.keys():
        out[label_map[label]["id"]] = str(label)
    return out


def make_color2label(label_map):
    out = {}
    for label in label_map.keys():
        out[label_map[label]["color"]] = str(label)
    return out


def make_id2color(label_map):
    out = {}
    for label in label_map.keys():
        out[out[label]["idx"]] = out[label]["color"]
    return out


def color_histogram(image_bgr):
    hist = {}
    for j in range(image_bgr.shape[0]):
        for i in range(image_bgr.shape[1]):
            pixel = tuple(image_bgr[j, i, :])
            if pixel not in hist.keys():
                hist[pixel] = 0
            hist[pixel] += 1
    return hist

