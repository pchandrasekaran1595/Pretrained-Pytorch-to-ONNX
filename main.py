import os
import sys

from torch import randn, onnx
from models import get_model

import warnings
warnings.filterwarnings("ignore")


SAVE_PATH = "output"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

NAMES = ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
         "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wresnet50", "wresnet101", "resnext50", "resnext101",
         "densenet121", "densenet161", "densenet169", "densenet201", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
         "f_resnet50", "f_mobilenet", "f_mobilenet_320", "retinanet", "ssd300", "ssdlite", "m_resnet50", 
         "fcn_resnet50", "fcn_resnet101", "dl_resnet50", "dl_resnet101", "dl_mobile", "lraspp"]


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def main():

    args_1: tuple = ("--name", "-n")
    args_2: tuple = ("--size", "-s")
    args_3: str = "--opver"

    name: str = None
    size: int = 224
    opset_version: int = 9

    if args_1[0] in sys.argv: name = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: name = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: size = int(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv: size = int(sys.argv[sys.argv.index(args_2[1]) + 1])

    if args_3 in sys.argv: opset_version = int(sys.argv[sys.argv.index(args_3) + 1])

    assert name is not None and name in NAMES, "Enter a valid model name"

    batch_size = 1
    model = get_model(name)
    dummy = randn(batch_size, 3, size, size)

    breaker()
    print("Exporting Model ....")

    onnx.export(model=model, 
                args=dummy, 
                f=os.path.join(SAVE_PATH, "{}.onnx".format(name)), 
                input_names=["input"], 
                output_names=["output"], 
                opset_version=opset_version,
                export_params=True,
                training=onnx.TrainingMode.EVAL,
                dynamic_axes={
                    "input"  : {0 : "batch_size"},
                    "output" : {0 : "batch_size"}
                }
    )

    breaker()
    print("Export Complete")
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
