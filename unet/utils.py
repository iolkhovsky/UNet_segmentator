from transform_utils import *
import cv2


COLORS = (
    (0, 0, 128),
    (0, 128, 0),
    (0, 128, 128),
    (128, 0, 0),
    (128, 0, 128),
    (128, 128, 0),
    (128, 128, 128),
    (0, 0, 64),
    (0, 0, 192),
    (0, 128, 64),
    (0, 128, 192),
    (128, 0, 64),
    (128, 0, 192),
    (128, 128, 64),
    (128, 128, 192),
    (0, 64, 0),
    (0, 64, 128),
    (0, 192, 0),
    (0, 192, 128),
    (128, 64, 0)
)


def decode_prediction(pred, to_tensors):
    if pred.shape[0] > len(COLORS):
        raise RuntimeError("Output channels count too high")
    idmap = np.argmax(pred, axis=0)
    return decode_target(idmap, to_tensors)


def decode_target(target, to_tensors):
    out = np.zeros(shape=(target.shape[0], target.shape[1], 3), dtype=np.uint8)
    for j in range(out.shape[0]):
        for i in range(out.shape[1]):
            out[j, i] = COLORS[target[j, i]]
    if to_tensors:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = array_yxc2cyx(out)
        out = torch.from_numpy(out)
    return out


def decode_input_tensor(img, to_tensors):
    image = denormalize_img_cyx(img)
    image = array_cyx2yxc(image)
    image = array_yxc2cyx(image)
    if to_tensors:
        image = torch.from_numpy(image)
    return image


def visualize_prediction_target(src_batch, pred_batch, target_batch, to_tensors=True):
    if type(pred_batch) == torch.Tensor:
        pred_batch = pred_batch.detach().numpy()
    if type(target_batch) == torch.Tensor:
        target_batch = target_batch.detach().numpy()
    if type(src_batch) == torch.Tensor:
        src_batch = src_batch.detach().numpy()

    src_imgs, out_pred, target_pred = [], [], []
    for pred in pred_batch:
        out_pred.append(decode_prediction(pred, to_tensors))
    for target in target_batch:
        target_pred.append(decode_target(target, to_tensors))
    for src in src_batch:
        src_imgs.append(decode_input_tensor(src, to_tensors))
    return src_imgs, out_pred, target_pred


def get_accuracy(pred_batch, target_batch):
    if type(pred_batch) == torch.Tensor:
        pred_batch = pred_batch.detach().numpy()
    if type(target_batch) == torch.Tensor:
        target_batch = target_batch.detach().numpy()
    idmap = np.argmax(pred_batch, axis=1)
    total = target_batch.shape[0] * target_batch.shape[1] * target_batch.shape[2]
    return np.sum((idmap == target_batch).astype(np.uint8)) / (1. * total)
