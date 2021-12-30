## Pytorch to ONNX Converter

- Python Script to convert Pretrained Pytorch Models to ONNX Format

- Command Line Arguments
    1. `--name | -n`
    2. `--size | -s`
    3. `--opver`

- Run using `python main.py <args>`

- Supported Names

`alexnet, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,` <br>
`resnet18, resnet34, resnet50, resnet101, resnet152, wresnet50, wresnet101, resnext50, resnext101,` <br>
`densenet121, densenet161, densenet169, densenet201, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,` <br>
`f_resnet50, f_mobilenet, f_mobilenet_320, retinanet, ssd300, ssdlite, m_resnet50,` <br>
`fcn_resnet50, fcn_resnet101, dl_resnet50, dl_resnet101, dl_mobile, lraspp`

<br>

*Note: Can convert mobilenet models to ONNX as well, but doesn't work when used to perform inference since it cannot create some layers.* 