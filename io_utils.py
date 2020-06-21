import torch
import datetime


MODELS_REPOSITORY_PATH = "/home/igor/models_checkpoints/"
CUDA_IS_AVAILABLE = torch.cuda.is_available()


def get_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = stamp.replace(" ", "_")
    stamp = stamp.replace(":", "_")
    stamp = stamp.replace("-", "_")
    return stamp


def get_readable_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return stamp


def compile_checkpoint_name(name, hint=None):
    name = name.replace(" ", "_")
    file_name = MODELS_REPOSITORY_PATH
    if hint is not None:
        file_name += hint+"_"
    file_name += name + "_"
    file_name += get_timestamp()
    file_name += ".torchmodel"
    return file_name


def load_model(path, logger=None):
    if MODELS_REPOSITORY_PATH not in path:
        if logger:
            logger("Warning: Wrong path to models repository -> adding prefix automatically")
    path = MODELS_REPOSITORY_PATH + path
    if logger:
        logger("Loading model: "+path)
    torch_model = torch.load(path)
    return torch_model


def save_model(torch_model, use_cuda, hint=None, logger=None):
    fname = compile_checkpoint_name(str(torch_model), hint)
    if CUDA_IS_AVAILABLE and use_cuda:
        torch_model = torch_model.cpu()
    torch.save(torch_model, fname)
    if logger:
        logger("Model '"+fname+"' successfully saved.")
    if CUDA_IS_AVAILABLE and use_cuda:
        torch_model = torch_model.cuda()
    return torch_model
