import cv2
import argparse

from transform_utils import *
from io_utils import load_model
from unet.utils import *
from voc_dataset.utils import *


def make_tensor_for_net(cv_img, in_img_sz=(572, 572)):
    image = cv2.resize(cv_img, in_img_sz)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = array_yxc2cyx(image)  # replace channels position - YXC to CYX
    image = normalize_img_cyx(image)
    in_tensor = torch.from_numpy(image).reshape(1, 3, in_img_sz[0], in_img_sz[1])
    return in_tensor


model_path = "e26b199_UNet_2020_06_28_10_52_50.torchmodel"

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to model's checkpoint to load")
args = parser.parse_args()

if args.model:
    model_path = args.model
print("Loaded model from: ", model_path)

model = load_model(model_path, logger=print)
model.eval()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    src_frame_sz = frame.shape
    img_tensor = make_tensor_for_net(frame.copy())
    prediction = model.forward(img_tensor).detach().numpy()[0]
    prediction = decode_prediction(prediction, to_tensors=False)
    cv2.imshow("Stream", frame)
    cv2.imshow("Processed", prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()