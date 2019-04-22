"""Yay."""
import numpy as np
import torch

from ConvertModel import ConvertModel_ncnn
from models.DeepBilateralNetCurves import DeepBilateralNetCurves

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def ncnn_convert():
    lowres = [256, 256]
    fullres = [512, 512]
    luma_bins = 8
    spatial_bins = 16
    channel_multiplier = 1
    guide_pts = 16
    input_shape = [1, 3, *lowres]
    pytorch_net = DeepBilateralNetCurves(lowres,
                                         luma_bins,
                                         spatial_bins,
                                         channel_multiplier,
                                         guide_pts)
    pytorch_net = pytorch_net.eval()
    pytorch_net.load_state_dict(
        torch.load(
            './model_latest.pth',
            map_location=lambda storage, loc: storage))
    text_net, binary_weights = ConvertModel_ncnn(pytorch_net, input_shape, softmax=False)
    with open('./hdrnet_guidemap.param', 'w') as f:
        f.write(text_net)
    with open('./hdrnet_guidemap.bin', 'w') as f:
        for weights in binary_weights:
            for blob in weights:
                blob_32f = blob.flatten().astype(np.float32)
                blob_32f.tofile(f)


if __name__ == '__main__':
    ncnn_convert()
