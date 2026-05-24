import os
from collections import namedtuple
import numbers
import math
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


import cv2 as cv
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

WEIGHTS_PATH = MODELS_DIR / "DMAGNet_model23HR_Finale_ft.pt"

for path in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "DMAGNET"
PRETRAINED_WEIGHTS = "GALAXYZOO"

GALAXYZOO_MEAN = np.array([0.167970, 0.163042, 0.159367], dtype=np.float32)
GALAXYZOO_STD  = np.array([0.106749, 0.101212, 0.091870], dtype=np.float32)

###################################################
## DMAGNet ########################################
###################################################

class Stem(nn.Module):

    """
    The input image is first processed by a two-dimensional convolution with a 7x7 receptive field.
    This convolution increases the number of feature channels while simultaneously reducing the spatial resolution,
    allowing the network to capture coarse, low-level visual patterns
    such as edges and textures early in the architecture.

    The convolutional output is then normalized using batch normalization,
    which stabilizes training by reducing internal covariate shift and enabling higher learning rates.

    A rectified linear unit (ReLU) activation function is subsequently applied
    to introduce non-linearity into the model, allowing it to learn more expressive feature representations.

    Finally, a max-pooling operation further reduces the spatial resolution of the feature maps
    by retaining only the strongest activations within local neighborhoods.
    This step improves computational efficiency and introduces a degree of translation invariance.
    """

    def __init__(self, in_channels=3, out_channels=64):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class DLK(nn.Module):
    """

    The input is processed in three parallel branches, each operating at a different effective receptive field:

        - Large-kernel branch: a channel-reduced projection followed by a depthwise 5×5 convolution,
          then expanded back to C channels
        - Medium-kernel branch: analogous structure with a depthwise 3×3 convolution.
        - Small-kernel branch: a lightweight 1×1 convolutional pathway without spatial convolution.

    In all branches, pointwise convolutions perform channel reduction and expansion,
    with the reduced dimensionality given by C / reduction_ratio. Batch normalization and
    ReLU activations are applied after each convolutional stage.

    The outputs of the three branches are concatenated along the channel dimension, yielding a tensor of shape

    R^{BATCH x 3 x CHANNELS x HEIGHT x WIDTH}

    A channel-wise attention mechanism is then applied: global average pooling aggregates spatial information,
    followed by a bottleneck MLP implemented via 1×1 convolutions.
    A sigmoid activation produces per-branch, per-channel attention weights,
    which are used to adaptively reweight the outputs of the three branches.

    The final output is obtained as a weighted sum of the three branch outputs,
    followed by a residual addition with the original input, x:

    y = sum_{i \in \{ small, medium, large \} } weights_i * f_i(x) + x

    The output tensor has the same shape as the input:

    y \in R^{BATCH x CHANNELS x HEIGHT x WIDTH}

    """
    def __init__(self, channels, reduction_ratio=4):
        super(DLK, self).__init__()
        self.channels = channels
        reduced_channels = max(1, channels // reduction_ratio)

        self.large_kernel = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=5,
                      padding=2, dilation=1, groups=reduced_channels, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.medium_kernel = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3,
                      padding=1, groups=reduced_channels, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.small_kernel = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        large = self.large_kernel(x)
        medium = self.medium_kernel(x)
        small = self.small_kernel(x)
        combined = torch.cat([large, medium, small], dim=1)
        attention_weights = self.attention(combined)
        large_w, medium_w, small_w = torch.split(attention_weights, self.channels, dim=1)
        out = large * large_w + medium * medium_w + small * small_w

        return out + identity


class MS_FFN(nn.Module):

    """

  The input is first projected into a higher-dimensional feature space via a pointwise (1×1) convolution,
  #expanding the channel dimension to C x expansion_ratio. This is followed by batch normalization
  and a ReLU activation.

  The expanded features are then processed by two parallel depthwise convolutions:
      - a standard 3×3 depthwise convolution capturing local spatial context,
      - a dilated 3×3 depthwise convolution (dilation = 2) capturing a larger receptive field.

  The outputs of the two depthwise paths are concatenated along the channel dimension,
  producing a tensor of shape

  R^{BATCH x 2 x CHANNELS x expansion_ratio x HEIGHT x WIDTH}

  Batch normalization and ReLU activation are applied to the concatenated features,
  which are then projected back to the original channel dimension via a second pointwise convolution.

  The final output is obtained through a residual addition with the input:

  y = F(x) + x,
  where F() denotes the multi-scale feed-forward transformation.

  y \in R^{BATCH x CHANNELS x HEIGHT x WIDTH}

  """
    def __init__(self, channels, expansion_ratio=4):
        super(MS_FFN, self).__init__()
        hidden_channels = channels * expansion_ratio

        self.conv1 = nn.Sequential(
          nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
      )
        self.depthwise1 = nn.Conv2d(hidden_channels, hidden_channels, 3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.depthwise2 = nn.Conv2d(hidden_channels, hidden_channels, 3,
                                  padding=2, dilation=2, groups=hidden_channels, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
          nn.Conv2d(hidden_channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
      )

    def forward(self, x):
        identity = x
        x = self.conv1(x)

        x1 = self.depthwise1(x)
        x2 = self.depthwise2(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        return x + identity


class AFF(nn.Module):
    """
    Global spatial information is first aggregated via adaptive average pooling,
    producing a channel descriptor of shape R^{BATCH x CHANNEL x 1 x 1}

    This descriptor is passed through a lightweight bottleneck transformation
    implemented with two pointwise (1×1) convolutions. The first reduces the channel dimension
    to CHANNELS / 4 and applies a ReLU activation, while the second restores the original channel dimensionality.
    A sigmoid activation generates channel-wise attention weights in the range [0,1].

    The input feature map is then modulated by these weights through element-wise multiplication,
    enabling selective emphasis or suppression of channels.

    The final output is obtained by adding the attention-modulated features to the original input, x,  via a residual connection:

    y = x * attention + x

    The output tensor has the shape as the input

    y \in R^{BATCH x CHANNELS x HEIGHT x WIDTH}

    """
    def __init__(self, channels):
        super(AFF, self).__init__()

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        channel_weights = self.channel_att(x)
        out = x * channel_weights
        return out + identity


class DMA_Block(nn.Module):
    """

    A residual shortcut path is first constructed.
    If spatial downsampling is required or the input and output channel dimensions differ,
    the shortcut applies a 1×1 convolution with stride 2 (if downsampling) followed by batch normalization;
    otherwise, an identity mapping is used.

    The main branch follows a pre-activation design. The input is first batch-normalized and passed through a ReLU activation,
    then processed by a 3×3 convolution that optionally performs spatial downsampling via stride 2
    and projects the features to out_channels channels.

    The resulting feature map is further normalized and activated before being sequentially processed
    by the three specialized modules

        - DLK
        - MS-FFN
        - AFF

    The output of the main branch is combined with the shortcut through residual addition:

    y = F(x) + S(x),

    where F denotes the composite transformation and S the shortcut mapping.

    The output tensor has shape:

    y \in R^{BATCH x out_channels x H' x W'},

    where H' = H/2 and W' = W/2 if downsampling is enabled;
    H' = H and W' = W otherwise

    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(DMA_Block, self).__init__()

        self.downsample = downsample
        self.stride = 2 if downsample else 1

        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dlk = DLK(out_channels)
        self.ms_ffn = MS_FFN(out_channels)
        self.aff = AFF(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.dlk(out)
        out = self.ms_ffn(out)
        out = self.aff(out)

        return out + identity


class LAM(nn.Module):

    """

  The input is first processed by a 3×3 convolution that reduces the channel dimensionality to out_channels/2,
  followed by batch normalization and ReLU activation. This step performs local feature extraction while compressing the channel space.

  A channel-wise attention mechanism is then applied to the intermediate features. Two successive pointwise (1×1) convolutions
  implement a bottleneck transformation that first reduces the channel dimension to out_channels/4 and then restores
  it to out_channels / 2. A sigmoid activation produces channel attention weights,
  which modulate the intermediate feature map via element-wise multiplication.

  The attention-refined features are finally passed through a second 3×3 convolution that expands the channel dimension
  to out_channels, followed by batch normalization and ReLU activation.

  The output of the block is a tensor of shape

  y \in R^{BATCH x out_channels x HEIGHT x WIDTH}

  No residual connection is used; the block performs a pure feature transformation with integrated attention.

  """
    def __init__(self, in_channels, out_channels):
        super(LAM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        att = self.attention(x)
        x = x * att
        x = self.conv2(x)
        return x


class AttentionPooling(nn.Module):
    """

    A spatial attention map is first computed from the input
    using a lightweight convolutional subnetwork composed of two pointwise (1×1) convolutions
    with an intermediate channel reduction to CHANNELS / 8 and a ReLU activation.

    A sigmoid function produces a single-channel attention map of shape R^{BATCH x 1 x HEIGHT x WIDTH}
    encoding the relative importance of each spatial location.

    The input feature map is modulated by this attention map via element-wise multiplication,
    emphasizing informative regions and suppressing less relevant spatial responses.

    The attention-weighted features are then aggregated using adaptive global average pooling,
    which reduces the spatial dimensions to 1x1 independently of the input resolution.

    The pooled features are flattened to produce an output tensor of shape

    y \in R^{BATCH x CHANNEL}

    This vector representation can be directly used for classification or further fully connected processing.


    """
    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_weights = self.attention(x)
        out = x * att_weights
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


class Head(nn.Module):
    """

    The input feature map is first refined by a 3×3 convolution that reduces the channel dimensionality to 256,
    followed by batch normalization and ReLU activation. This step consolidates high-level features
    while preserving spatial resolution.

    The refined feature map is then processed by an attention-based pooling layer,
    which computes a spatial attention map, applies attention-weighted feature modulation,
    and performs adaptive global average pooling. This produces a compact, fixed-length
    representation that is independent of the input spatial dimensions.

    The resulting feature vector is passed through a fully connected classification module,
    consisting of a linear projection to a 128-dimensional hidden space, ReLU activation,
    dropout regularization, and a final linear layer that maps to num_classes output logits.

    The output is a tensor

    y \in R^{BATCH x num_classes}

    representing the unnormalized class scores for each input sample

    """
    def __init__(self, in_channels=1024, num_classes=10):
        super(Head, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.attention_pool = AttentionPooling(256)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.attention_pool(x)
        x = self.fc(x)
        return x


class DMAGNet(nn.Module):
    """

    The network begins with a stem module,
    which applies strided convolution and max pooling to rapidly reduce the spatial resolution
    while expanding the channel dimension from 3 to 64.

    The core of the network consists of three hierarchical stages built from DMA blocks:

    Layer 1: two DMA blocks operating at 64×64 resolution.
    The first block performs spatial downsampling and increases the channel dimension from 64 to 128,
    while the second preserves resolution and channel size.

    Layer 2: a single DMA block that downsamples the feature maps to 16×16 and
    increases the channel dimension from 128 to 256.

    Layer 3: two DMA blocks operating at 16×16 and 8×8 resolutions.
    The first block downsamples and expands channels from 256 to 512,
    while the second refines features at constant resolution.

    Following the hierarchical stages,
    a Lightweight Attention Module (LAM) further processes the 512-channel feature maps
    and expands them to 1024 channels while preserving spatial resolution.

    The final classification head refines high-level features,
    applies attention-based global pooling to produce a fixed-length representation,
    and maps it to class logits via a fully connected classifier.

    The network outputs a tensor

    y \in R^{BATCH x num_classes}

    representing the unnormalized class scores for each input image.

    """
    def __init__(self, num_classes=10, input_size=256):
        super(DMAGNet, self).__init__()

        self.stem = Stem(in_channels=3, out_channels=64)

        self.layer1 = nn.Sequential(
            DMA_Block(64, 128, downsample=True),
            DMA_Block(128, 128, downsample=False)
        )

        self.layer2 = nn.Sequential(
            DMA_Block(128, 256, downsample=True),
        )

        self.layer3 = nn.Sequential(
            DMA_Block(256, 512, downsample=True),
            DMA_Block(512, 512, downsample=False)
        )

        self.lam = LAM(512, 1024)

        self.head = Head(1024, num_classes)

        self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.lam(x)

        x = self.head(x)

        return x
    
###################################################
## HELPERS ########################################
###################################################
    
"""

The function load_image loads an image from a specified filesystem path and returns a normalized RGB representation suitable for further processing.
The input img_path is a string indicating the path to the image file; if the path does not exist, the function raises an exception.
The image is read using OpenCV and converted from BGR to RGB format.

An optional argument, target_shape, controls image resizing. If target_shape is None, the image is returned at its original resolution.
If target_shape is an integer different from −1, it specifies the target width and the height is scaled proportionally to preserve the original aspect ratio.
If target_shape is a tuple, it specifies the target height and width explicitly, and the image is resized accordingly.

The image is then converted to float32 format and normalized to the [0,1] range.
The function returns a NumPy array of shape (H,W,3) containing the normalized RGB image.

"""
def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]

    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)


    img = img.astype(np.float32)
    img /= 255.0
    return img

"""

The function save_and_maybe_display_image saves an image to disk and optionally displays it.
The input dump_img must be a NumPy array representing an image, while config is a dictionary containing configuration parameters,
including the output directory and display settings. An optional argument, name_modifier, can be provided to control the output filename.

The function ensures that the output directory specified in config['dump_dir'] exists and determines the image filename either from name_modifier
or by constructing it using the configuration parameters.
If the input image is not in unsigned 8-bit integer format, it is scaled and converted to uint8. The image is then saved to disk in BGR format using OpenCV.

If the configuration flag config['should_display'] is enabled, the image is displayed using Matplotlib.
The function returns the filesystem path to the saved image.

"""
def save_and_maybe_display_image(config, dump_img, name_modifier=None):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    dump_dir = config['dump_dir']
    os.makedirs(dump_dir, exist_ok=True)

    if name_modifier is not None:
        dump_img_name = str(name_modifier).zfill(6) + '.jpg'
    else:
        dump_img_name = build_image_name(config)

    if dump_img.dtype != np.uint8:
        dump_img = (dump_img*255).astype(np.uint8)

    dump_path = os.path.join(dump_dir, dump_img_name)
    cv.imwrite(dump_path, dump_img[:, :, ::-1])

    if config['should_display']:
        fig = plt.figure(figsize=(7.5,5), dpi=100)
        plt.show()

    return dump_path


"""

The function build_image_name generates a descriptive filename for an output image based on the parameters stored in the input configuration dictionary config.
The filename encodes information about the input image source, the network layers used, image resolution, model type, pretrained weights,
Deep Dream pyramid settings, optimization parameters, and smoothing coefficients.

If the configuration specifies the use of random noise, a fixed identifier is used as the input name; otherwise,
the base name of the input image file is extracted. The function returns a string containing the constructed filename,
which uniquely reflects the experimental settings associated with the generated image.

"""

def build_image_name(config):
    input_name = 'rand_noise' if config['use_noise'] else config['input'].split('.')[0]
    layers = '_'.join(config['layers_to_use'])
    img_name = f'{input_name}_width_{config["img_width"]}_model_{config["model_name"]}_{config["pretrained_weights"]}_{layers}_pyrsize_{config["pyramid_size"]}_pyrratio_{config["pyramid_ratio"]}_iter_{config["num_gradient_ascent_iterations"]}_lr_{config["lr"]}_shift_{config["spatial_shift_size"]}_smooth_{config["smoothing_coefficient"]}.jpg'
    return img_name

"""

The function pre_process_numpy_img applies channel-wise normalization to an input image represented as a NumPy array.
The input img must be a NumPy array containing image data, while mean and std specify the normalization statistics.
The function subtracts the mean and divides by the standard deviation for each channel.
The output is a normalized NumPy array with the same shape as the input image.

"""
def pre_process_numpy_img(img, mean=GALAXYZOO_MEAN, std=GALAXYZOO_STD):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'
    img = (img - mean) / std
    return img

"""

The function post_process_numpy_img reverses a channel-wise normalization applied to an image represented as a NumPy array.
The input img must be a NumPy array, while mean and std specify the normalization statistics.
If the image is in channel-first format, it is converted to channel-last format.
The function rescales the image by multiplying by the standard deviation and adding the mean, then clips pixel values to the
[0,1] range. The output is a NumPy array containing the denormalized image.

"""
def post_process_numpy_img(img, mean=GALAXYZOO_MEAN, std=GALAXYZOO_STD):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, 2)
    mean = mean.reshape(1, 1, -1)
    std = std.reshape(1, 1, -1)
    img = (img * std) + mean
    img = np.clip(img, 0., 1.)
    return img

"""

The function pytorch_input_adapter converts an input image into a PyTorch tensor suitable for model input.
The input img is expected to be an image in NumPy array format. The function converts the image to a tensor, moves it to the specified computation device,
and adds a batch dimension. Gradient computation is enabled for the tensor. The function returns a PyTorch tensor of shape (1,C,H,W).

"""
def pytorch_input_adapter(img):
    tensor = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)
    tensor.requires_grad = True
    return tensor

"""

The function pytorch_output_adapter converts a PyTorch tensor into a NumPy array suitable for image processing.
The input tensor is expected to be a batched tensor. The function moves the tensor to the CPU, detaches it from the computation graph,
removes the batch dimension, and rearranges the axes from channel-first to channel-last format. The output is a NumPy array of shape
(H,W,C).

"""
def pytorch_output_adapter(tensor):
    return np.moveaxis(tensor.to('cpu').detach().numpy()[0], 0, 2)

"""

The function get_new_shape computes the spatial resolution of an image at a given level of a spatial pyramid.
The input original_shape specifies the height and width of the original image, while current_pyramid_level indicates the current level within the pyramid.
The function uses the pyramid scaling parameters stored in the configuration dictionary config to compute a scaled image shape based on an exponential ratio.

The function returns a NumPy array containing the new image height and width for the specified pyramid level.
If the computed resolution falls below a predefined minimum size, the function terminates execution to prevent invalid pyramid configurations.

"""
def get_new_shape(config, original_shape, current_pyramid_level):
    SHAPE_MARGIN = 10
    pyramid_ratio = config['pyramid_ratio']
    pyramid_size = config['pyramid_size']
    exponent = current_pyramid_level - pyramid_size + 1
    new_shape = np.round(np.float32(original_shape) * (pyramid_ratio**exponent)).astype(np.int32)

    if new_shape[0] < SHAPE_MARGIN or new_shape[1] < SHAPE_MARGIN:
        print(f'Pyramid size {config["pyramid_size"]} with pyramid ratio {config["pyramid_ratio"]} gives too small pyramid levels with size={new_shape}')
        print(f'Please change the parameters.')
        exit(0)

    return new_shape

"""

The function apply_sharpening applies an image sharpening operation to a normalized image represented as a NumPy array.
The input img_np is expected to contain pixel values in the [0,1] range.
The function converts the image to 8-bit format, applies Gaussian blurring, and combines the original image with the blurred version using weighted addition to enhance high-frequency details.
The output is a NumPy array in float32 format with pixel values rescaled to the [0,1] range.

"""
def apply_sharpening(img_np):
    img_cv = (img_np * 255).astype(np.uint8)
    gaussian = cv.GaussianBlur(img_cv, (0, 0), 2.0)
    sharp = cv.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)
    return sharp.astype(np.float32) / 255.0

"""

This function performs a circular spatial translation of an input tensor along the height and width dimensions.
The amount of vertical and horizontal shift is specified by integer offsets, which can be inverted when the should_undo flag is enabled.
The translation is implemented using a wrap-around operation so that spatial dimensions are preserved.
The operation is executed without gradient tracking, and the returned tensor is explicitly marked to require gradients.

The input consists of a 4D PyTorch tensor and two integer shift values, with an optional boolean flag to reverse the displacement.
The output is a tensor of identical shape, spatially shifted according to the specified offsets and ready for subsequent gradient-based optimization.

"""
def random_circular_spatial_shift(tensor, h_shift, w_shift, should_undo=False):
    if should_undo:
        h_shift = -h_shift
        w_shift = -w_shift
    with torch.no_grad():
        rolled = torch.roll(tensor, shifts=(h_shift, w_shift), dims=(2, 3))
        rolled.requires_grad = True
        return rolled

"""

The class CascadeGaussianSmoothing implements a multi-scale Gaussian smoothing operation applied channel-wise to an input tensor.
The module is initialized with a Gaussian kernel size and a base standard deviation sigma.
Three Gaussian kernels are constructed using scaled versions of the base standard deviation, forming a cascade of smoothing filters with increasing blur strength.

During initialization, each Gaussian kernel is generated analytically, normalized to unit sum,
reshaped for depthwise convolution, and replicated across input channels.
The kernels are stored as fixed convolution weights and applied using grouped convolution.

In the forward pass, the input tensor is first padded using reflection padding to preserve spatial resolution.
The three Gaussian filters are then applied independently to the input via depthwise convolution.
The final output is obtained by averaging the three smoothed tensors.
The module returns a tensor with the same shape as the input, representing the smoothed version of the original data.

"""
class CascadeGaussianSmoothing(nn.Module):

    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]

        self.pad = int(kernel_size[0] / 2)

        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            kernel = kernel / torch.sum(kernel)
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)
            kernel = kernel.to(DEVICE)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3

"""

This class defines a wrapper module around the DMAGNet architecture for feature extraction and interpretability analysis.
During initialization, a DMAGNet model configured for ten output classes and fixed input resolution is instantiated
and optionally initialized with pretrained GalaxyZoo weights loaded from disk. When gradient computation is disabled,
all model parameters are frozen and the network is set to evaluation mode. The class also defines an ordered list of layer identifiers
corresponding to the main architectural stages used for feature visualization.

The forward pass explicitly propagates the input tensor through the stem, successive DMA blocks, and the local attention module,
capturing the intermediate feature maps produced at each stage. These activations are returned as a named tuple,
preserving both their order and semantic association with specific network components, and enabling selective access to internal representations during downstream analysis.

"""
class DMAGNetExperimental(nn.Module):
    def __init__(self, weights_path=None, requires_grad=False):
        super().__init__()

        self.dmagnet = DMAGNet(num_classes=10, input_size=256)

        if weights_path:
            if os.path.exists(weights_path):
                try:
                    state_dict = torch.load(str(weights_path), map_location=DEVICE)
                    self.dmagnet.load_state_dict(state_dict)
                except RuntimeError as e:
                    print(f"Weights loading error...: {e}")
            else:
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

        self.layer_names = ['stem', 'layer1', 'layer2', 'layer3', 'lam']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
            self.dmagnet.eval()

    def forward(self, x):
        x = self.dmagnet.stem(x)
        stem_out = x
        x = self.dmagnet.layer1(x)
        layer1_out = x
        x = self.dmagnet.layer2(x)
        layer2_out = x
        x = self.dmagnet.layer3(x)
        layer3_out = x
        x = self.dmagnet.lam(x)
        lam_out = x

        DMAGNetOutputs = namedtuple("DMAGNetOutputs", self.layer_names)
        out = DMAGNetOutputs(stem_out, layer1_out, layer2_out, layer3_out, lam_out)
        return out


"""

This function instantiates a DMAGNetExperimental model and initializes it with pretrained GalaxyZoo weights loaded from a specified file path.
The function validates the existence of the weights file, constructs the model with gradient computation disabled,
transfers it to the configured computation device, and sets it to evaluation mode.

The input consists of a file path pointing to the serialized model weights.
The output is a fully initialized and device-ready DMAGNetExperimental model configured for inference and feature extraction.

"""
def fetch_and_prepare_model(weights_path):
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Invalid weights path: {weights_path}")

    model = DMAGNetExperimental(
        weights_path=weights_path,
        requires_grad=False
    ).to(DEVICE)

    model.eval()
    return model

def plot_deepdream_summary(config, image_output_dir):
    image_paths = [
        INPUT_DIR / config["input"],
        image_output_dir / "UltraSharp_stem.jpg",
        image_output_dir / "UltraSharp_layer1.jpg",
        image_output_dir / "UltraSharp_layer2.jpg",
        image_output_dir / "UltraSharp_layer3.jpg",
        image_output_dir / "UltraSharp_lam.jpg",
    ]

    titles = ["Input", "Stem", "DMA 1", "DMA 2", "DMA 3", "LAM"]

    images = []
    for path in image_paths:
        img = cv.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    plt.figure(figsize=(18, 4))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=11)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
def build_config(input_image="spiral_galaxy.png"):
    return {
        "input": os.path.basename(input_image),

        "model_name": MODEL_NAME,
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "weights_path": WEIGHTS_PATH,

        "pyramid_size": 2,
        "pyramid_ratio": 1.3,
        "smoothing_coefficient": 0.5,
        "use_noise": False,
        "should_display": False,

        "dump_dir": OUTPUT_DIR,
    }


def run_deepdream_for_layer(config, model, layer):
    print(f"\n--- Processing layer: {layer} ---")
    config["layers_to_use"] = [layer]

    config.update({
        "img_width": 256,
        "num_gradient_ascent_iterations": 20,
        "lr": 0.05,
        "spatial_shift_size": 10,
    })
    img = deep_dream_static_image(config, model)

    config.update({
        "img_width": 512,
        "num_gradient_ascent_iterations": 10,
        "lr": 0.03,
        "spatial_shift_size": 5,
    })
    img = cv.resize(img, (512, 512), interpolation=cv.INTER_LANCZOS4)
    img = deep_dream_static_image(config, model, img=img)

    config.update({
        "img_width": 1024,
        "num_gradient_ascent_iterations": 5,
        "lr": 0.01,
        "spatial_shift_size": 5,
    })
    img = cv.resize(img, (1024, 1024), interpolation=cv.INTER_LANCZOS4)
    img = deep_dream_static_image(config, model, img=img)

    img = apply_sharpening(img)

    save_path = save_and_maybe_display_image(
        config,
        img,
        name_modifier=f"UltraSharp_{layer}"
    )

    print(f"Saved -> {save_path}")
    return save_path


###################################################
## DEEP DREAM #####################################
###################################################

'''
What "mean squared activation magnitude" means


When an image is passed through a neural network, each layer produces an activation tensor. This tensor contains numerical
values that represent how strongly each neuron responds to the input.

The activation magnitude refers to the numerical size of these activation values. Large values mean that the layer is responding
strongly to the image; small values mean weak response.

To quantify this response, the activation tensor is compared to a tensor of zeros using the mean squared error. This operation
computes the average of the squared activation values across all spatial locations and channels

Mathematically, this is equivalent to computing the mean of the squared actiavtions:

\[

\frac{1}{N} \sum_{i} a_{i]^{2}

\]

where $ a_{i} $ are the activation and $ N $ is the total number of elements of the activation tensor.


Why is this used in Deep Dream ?


Maximizing the mean squared activation magnitude encourages the optimization to increase the overall strength of the activations
in the selected layer, regardless of their sign or spatial location.

This avoids cancelation effects between positive and negative activations and provides a smooth, well-behaved objective function
for gradient ascent.

As a result, the input image is modified so that it inccreasingly contains patterns that strongly excite the chosen netwrok layer.
'''

LOWER_IMAGE_BOUND = torch.tensor((-GALAXYZOO_MEAN / GALAXYZOO_STD).reshape(1, -1, 1, 1)).to(DEVICE)
UPPER_IMAGE_BOUND = torch.tensor(((1 - GALAXYZOO_MEAN) / GALAXYZOO_STD).reshape(1, -1, 1, 1)).to(DEVICE)

"""

The function gradient_ascent performs a single iteration of gradient ascent on an input image tensor in order to maximize the activations of specified network layers.
The input input_tensor is a PyTorch tensor with enabled gradient computation, while model is a neural network with fixed parameters.
The argument layer_ids_to_use specifies which layer activations are considered in the optimization, and iteration indicates the current optimization step.

The function computes the forward pass of the model and extracts the selected layer activations.
A loss is defined as the mean squared activation magnitude and backpropagated to obtain gradients with respect to the input tensor.
The resulting gradients are smoothed using a multi-scale Gaussian filter, normalized to zero mean and unit variance, and scaled by a learning rate specified in the configuration.
The input tensor is then updated in the direction of the processed gradient. Finally, gradients are cleared and the input values are clamped to predefined bounds.

"""

def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    out = model(input_tensor)


    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]

    losses = []
    for layer_activation in activations:
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    loss.backward()


    grad = input_tensor.grad.data

    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)

    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    input_tensor.data += config['lr'] * smooth_grad

    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


"""

The function deep_dream_static_image generates a Deep Dream visualization for a given neural network and set of target layers.
The input config is a configuration dictionary specifying model parameters, optimization settings, normalization statistics, and Deep Dream hyperparameters.
An optional input img may be provided as a NumPy array; if not, the image is loaded from disk according to the configuration.

The function initializes the specified pretrained model and identifies the target layers whose activations will be maximized.
The input image is normalized using dataset-specific statistics and processed through a multi-level spatial pyramid.
At each pyramid level, the image is resized, converted to a PyTorch tensor, and iteratively updated using gradient ascent on the selected layer activations,
with optional random spatial shifts applied to improve invariance.

After completing the optimization across all pyramid levels,
the function applies inverse normalization and returns a NumPy array representing the final Deep Dream image with pixel values in the [0,1] range.

"""
def deep_dream_static_image(config, model, img=None):


    try:
        layer_ids_to_use = [
            model.layer_names.index(layer_name)
            for layer_name in config["layers_to_use"]
        ]
    except ValueError:
        print(f"Invalid layer names: {config['layers_to_use']}")
        print(f"Available layers: {model.layer_names}")
        return None

    current_mean = GALAXYZOO_MEAN
    current_std = GALAXYZOO_STD

    if img is None:
        img_path = INPUT_DIR / config["input"]
        img = load_image(str(img_path), target_shape=config["img_width"])

        if config.get("use_noise", False):
            img = np.random.uniform(
                low=0.0, high=1.0, size=img.shape
            ).astype(np.float32)

    img = pre_process_numpy_img(img, mean=current_mean, std=current_std)
    original_shape = img.shape[:-1]

    for pyramid_level in range(config["pyramid_size"]):
        new_shape = get_new_shape(config, original_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = pytorch_input_adapter(img)

        for iteration in range(config["num_gradient_ascent_iterations"]):
            h_shift, w_shift = np.random.randint(
                -config["spatial_shift_size"],
                config["spatial_shift_size"] + 1,
                size=2
            )

            input_tensor = random_circular_spatial_shift(
                input_tensor, h_shift, w_shift
            )

            gradient_ascent(
                config, model, input_tensor, layer_ids_to_use, iteration
            )

            input_tensor = random_circular_spatial_shift(
                input_tensor, h_shift, w_shift, should_undo=True
            )

        img = pytorch_output_adapter(input_tensor)

    return post_process_numpy_img(img, mean=current_mean, std=current_std)

###################################################
## MAIN ###########################################
###################################################


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    model = fetch_and_prepare_model(WEIGHTS_PATH)

    input_images = sorted(
        [
            path for path in INPUT_DIR.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    if len(input_images) == 0:
        raise FileNotFoundError(f"No input images found in: {INPUT_DIR}")

    layers_to_test = ["stem", "layer1", "layer2", "layer3", "lam"]

    for input_image in input_images:
        print(f"\n==============================")
        print(f"Processing image: {input_image.name}")
        print(f"==============================")

        image_name = input_image.stem
        image_output_dir = OUTPUT_DIR / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        config = build_config(input_image=input_image.name)
        config["dump_dir"] = image_output_dir

        for layer in layers_to_test:
            try:
                run_deepdream_for_layer(config, model, layer)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU out of memory while processing {input_image.name} - {layer}.")
                    torch.cuda.empty_cache()
                else:
                    raise e

        plot_deepdream_summary(config, image_output_dir)

    print("\nAll Deep Dream images generated.")


if __name__ == "__main__":
    main()