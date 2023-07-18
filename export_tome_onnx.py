
import torch
import timm
import tome
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

model_name = "vit_base_patch16_224"
model = timm.create_model(model_name, pretrained=True)
input_size = model.default_cfg["input_size"][1]
transform = transforms.Compose([
     transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
     transforms.CenterCrop(input_size),
     transforms.ToTensor(),
     transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
])
img = Image.open("husky.png")
img_tensor = transform(img)[None, ...]
img_tensor = img_tensor.cuda()

device = "cuda:0"
runs = 50
batch_size = 256  # Lower this if you don't have that much memory
input_size = model.default_cfg["input_size"]

# Baseline benchmark
baseline_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)


tome.patch.timm(model)
model.r = 16
tome_throughput = tome.utils.benchmark(
    model,
    device=device,
    verbose=True,
    runs=runs,
    batch_size=batch_size,
    input_size=input_size
)



print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")



torch.onnx.export(
     model,
             img_tensor,
             f=f"vit_tome_imagenet.onnx",
             input_names=['image'],
             output_names=['logits'],
             do_constant_folding=True,
             opset_version=13,
       )
import onnxruntime as ort
import numpy as np
ort_sess = ort.InferenceSession(f"vit_tome_imagenet.onnx", providers=['CUDAExecutionProvider'])
outputs = ort_sess.run(None, {'image': img_tensor.cpu().numpy()})
pred = outputs[0][0].argmax(0)
print(pred)

