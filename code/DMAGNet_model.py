import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DOWNLOAD_PATH = RAW_DATA_DIR / "Galaxy10_DECals.h5"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
TEST_DIR = PROCESSED_DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DIR, TEST_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 0


########################################################
## DATA ################################################
########################################################


classes = [
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge"
]

mean=(0.16796959936618805, 0.16304244101047516, 0.15936698019504547)
std=(0.11394393444061279, 0.10874853283166885, 0.10070434212684631)
stats = (mean, std)

#Train set data augmentation.
#To allow the model to better generalize, some data augmentation is applied on the training set.
#Uniform random rotation in [-180°, 180°];
#Vertical and horizontal random flips
#mild random perturbations of brightness and contrast
train_transforms = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


#get_data_iterators()

"""

#The function first loads the full training dataset twice using ImageFolder:
#once with data augmentation enabled and once with only deterministic preprocessing.
#The test dataset is loaded separately using only deterministic preprocessing.

#The training dataset is then randomly split into two disjoint subsets using a fixed random seed
#to ensure reproducibility. Ninety percent of the samples are assigned to the training set,
#while the remaining ten percent are assigned to the validation set.

#Training samples are drawn from the augmented version of the dataset,
#while validation samples are drawn from the clean (non-augmented) version.
#This ensures that data augmentation is applied only during training and not during validation or testing.

#For each subset (training, validation, and test), a PyTorch DataLoader is created.
#The training loader shuffles the data at each epoch, while the validation and test loaders preserve a fixed order.
#Parallel data loading is enabled through multiple worker processes, and pinned memory is used to speed up data transfer to the GPU.

#OUTPUT:

The function returns three PyTorch data iterators:

    - train_iterator: a DataLoader that yields shuffled batches of augmented training images and their corresponding class labels.

    - valid_iterator: a DataLoader that yields batches of clean validation images and their corresponding class labels, without shuffling.

    - test_iterator: a DataLoader that yields batches of clean test images and their corresponding class labels, without shuffling.

Each iterator returns tuples of the form (images, labels), where:

    - images is a tensor of shape (BATCH_SIZE, C, IMG_SIZE, IMG_SIZE),

    - labels is a tensor containing the corresponding class indices.

"""


def get_data_iterators():
    print(f"Loading dataset (IMG_SIZE: {IMG_SIZE}x{IMG_SIZE})...")
    full_train_ds_aug = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    full_train_ds_clean = datasets.ImageFolder(root=TRAIN_DIR, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)

    num_train = len(full_train_ds_aug)
    indices = list(range(num_train))
    split = int(np.floor(0.10 * num_train))

    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(full_train_ds_aug, train_idx)
    val_subset = Subset(full_train_ds_clean, val_idx)

    print(f"Data loaded:")
    print(f" - Training (Augmented): {len(train_subset)}")
    print(f" - Validation:   {len(val_subset)}")
    print(f" - Test Set:             {len(test_dataset)}")


    train_iterator = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    valid_iterator = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    test_iterator = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )

    return train_iterator, valid_iterator, test_iterator

def prepare_imagefolder_dataset(
    h5_path=DOWNLOAD_PATH,
    output_dir=PROCESSED_DATA_DIR,
    test_size=3194,
    random_state=42
):
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Skip if already prepared
    if train_dir.exists() and test_dir.exists() and any(train_dir.iterdir()) and any(test_dir.iterdir()):
        print("Processed dataset already exists. Skipping preparation.")
        return

    print("Preparing ImageFolder dataset from .h5 file...")

    with h5py.File(h5_path, "r") as f:
        images = np.array(f["images"])
        labels = np.array(f["ans"])

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    for split_name, indices in [("train", train_idx), ("test", test_idx)]:
        for idx in indices:
            label = int(labels[idx])
            class_name = classes[label]

            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            img = Image.fromarray(images[idx].astype(np.uint8))
            img.save(class_dir / f"{idx:05d}.png")

    print("Dataset preparation completed.")


########################################################
## HELPERS #############################################
########################################################
    
def calculate_accuracy(y_pred, y):

    y_prob = F.softmax(y_pred, dim = -1)
    y_pred = y_pred.argmax(dim=1, keepdim = True)
    correct = y_pred.eq(y.view_as(y_pred)).sum()
    accuracy = correct.float()/y.shape[0]

    return accuracy


#the model loops over mini batches (x: images, y: labels) of the train iterator.
#x and y are copied to the same device where the model lives, so computations can happen there.
#1.   images are fed into the model, that produces a vector of scores probability for the classes.
#2.   the loss function is computed, as a single number that measures how wrong the predictins are compared to the ground truth
#     and we make this happen inside autocast, that basically speeds up the process allowing for the math to be made
#     in float16 (rather than slower float32), where it is safe to do so.
#3.   an accuracy score is computed
#4.   loss gradients are computed and multiplied by a big scaling factor (this is necessary, because, with float16,
#     gradients can become very small and underflow to zero). The scaling factor is updated at the end of each iteration as well.
#5.   gradients get scaled back to original size and clipped (max = 1.0) to avoid instability and divergence
#6.   the model's parameters get updated in the direction that reduces the loss
#7.   the learning rate gets updated

def train(model, iterator, optimizer, criterion, device, scaler, scheduler=None):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch_idx, (x, y) in enumerate(iterator):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            y_pred = model(x)
            loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)

        scaler.update()

        if scheduler:
            scheduler.step()

        if batch_idx % 10 == 0:
            print(
                f"Batch [{batch_idx+1}/{len(iterator)}] "
                f"- Loss: {loss.item():.4f} "
                f"- Acc: {acc.item()*100:.2f}%"
            )

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def model_training(n_epochs, model, train_iterator, valid_iterator, optimizer, criterion, device, scheduler=None, model_name='best_model.pt'):

    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_valid_loss = float('inf')

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device, scaler, scheduler)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)

        end_time = time.time()

        print(f"\nEpoch: {epoch+1}/{n_epochs} -- Time: {end_time-start_time:.2f} s")
        print(f"Train -- Loss: {train_loss:.3f}, Acc: {train_acc * 100:.2f}%")
        print(f"Val   -- Loss: {valid_loss:.3f}, Acc: {valid_acc * 100:.2f}%")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

    return train_losses, train_accs, valid_losses, valid_accs


def predict(model, iterator, device):

    model.eval()

    labels = []
    pred = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim=True)

            labels.append(y.cpu())
            pred.append(top_pred.cpu())

    labels = torch.cat(labels, dim=0)
    pred = torch.cat(pred, dim=0)

    return labels, pred


def model_testing(model, test_iterator, criterion, device, model_name='best_model.pt'):
    
    model.load_state_dict(torch.load(model_name, map_location=device))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f"Test -- Loss: {test_loss:.3f}, Acc: {test_acc * 100:.2f} %")
    
    
def plot_results(n_epochs, train_losses, train_accs, valid_losses, valid_accs):
    N_EPOCHS = n_epochs
    plt.figure(figsize=(20, 6))
    _ = plt.subplot(1,2,1)
    plt.plot(np.arange(N_EPOCHS)+1, train_losses, linewidth=3)
    plt.plot(np.arange(N_EPOCHS)+1, valid_losses, linewidth=3)
    _ = plt.legend(['Train', 'Validation'])
    plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Loss')

    _ = plt.subplot(1,2,2)
    plt.plot(np.arange(N_EPOCHS)+1, train_accs, linewidth=3)
    plt.plot(np.arange(N_EPOCHS)+1, valid_accs, linewidth=3)
    _ = plt.legend(['Train', 'Validation'])
    plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')
    
    
def print_report(model, test_iterator, device):
    labels, pred = predict(model, test_iterator, device)
    print(confusion_matrix(labels, pred))
    print("\n")
    print(classification_report(labels, pred))
    
    
########################################################
## DMAGNet #############################################
########################################################

#Stem:
#implements the initial feature extraction stage of a convolutional neural network. Its purpose is to transform the raw input image 
#into a lower-resolution but higher-dimensional feature representation that can be efficiently processed by deeper network layers.

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
    

#Dilated Large Kernel:
#implements a multi-branch, depthwise large-kernel convolution block with channel-wise attention and residual connection, 
#designed to capture multi-scale spatial context while preserving computational efficiency.

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
    
    
#Multi-Scale Feed-Forward Network:
#implements a multi-scale feed-forward convolutional block with channel expansion, parallel depthwise convolutions, and a residual 
#connection. It serves as a spatially aware alternative to the standard feed-forward network used in transformer-like architectures.

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
    
#Attention Feature Fusion:
#implements a channel-wise attention mechanism with residual fusion, designed to adaptively recalibrate feature responses while 
#preserving the original signal.

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
    
#DMA:
#implements a residual downsampling and multi-attention processing block, combining normalization-first pre-activation, multi-scale 
#spatial modeling, feed-forward enhancement, and attention-based feature fusion within a unified residual framework.

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
    
#Local Attention Module:
#implements a channel-attentive feature transformation block, designed to refine intermediate representations through 
#dimensionality reduction, attention-based modulation, and feature expansion.

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
    
#Attention Pooling:
#implements an attention-weighted global pooling mechanism, designed to aggregate spatial features into a compact vector while 
#preserving spatial saliency information.
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
    
#Head:
#implements the final classification head of the network, combining convolutional feature refinement, attention-based global 
#pooling, and a fully connected classifier.

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
    
    
#DMAGNet:
#a hierarchical convolutional neural network designed for image classification, integrating residual downsampling blocks, 
#multi-scale spatial processing, attention mechanisms, and attention-based global aggregation.

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
    

########################################################
## TRAINING ############################################
########################################################


def main():
    prepare_imagefolder_dataset()

    train_iterator, valid_iterator, test_iterator = get_data_iterators()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)

    model = DMAGNet(num_classes=10, input_size=256).to(device)

    # =========================
    # TRAINING
    # =========================

    N_EPOCHS = 140
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_iterator),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )

    first_model_path = MODELS_DIR / "DMAGNet_model23HR_Finale.pt"

    print(f"\nStarting DMAGNet training for {N_EPOCHS} epochs...")

    train_losses, train_accs, valid_losses, valid_accs = model_training(
        N_EPOCHS,
        model,
        train_iterator,
        valid_iterator,
        optimizer,
        criterion,
        device,
        scheduler=scheduler,
        model_name=first_model_path
    )

    plot_results(N_EPOCHS, train_losses, train_accs, valid_losses, valid_accs)

    model_testing(model, test_iterator, criterion, device, model_name=first_model_path)
    print_report(model, test_iterator, device)

    # =========================
    # FINE-TUNING
    # =========================

    model = DMAGNet(num_classes=10, input_size=256).to(device)
    model.load_state_dict(torch.load(first_model_path, map_location=device))

    EXTRA_EPOCHS = 40
    NEW_LEARNING_RATE = 1e-4

    optimizer = optim.AdamW(
        model.parameters(),
        lr=NEW_LEARNING_RATE,
        weight_decay=1e-2
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=EXTRA_EPOCHS,
        steps_per_epoch=len(train_iterator),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )

    fine_tuned_model_path = MODELS_DIR / "DMAGNet_model23HR_Finale_ft.pt"

    print(f"\nDMAGNet fine-tuning for {EXTRA_EPOCHS} more epochs...")

    train_losses, train_accs, valid_losses, valid_accs = model_training(
        EXTRA_EPOCHS,
        model,
        train_iterator,
        valid_iterator,
        optimizer,
        criterion,
        device,
        scheduler=scheduler,
        model_name=fine_tuned_model_path
    )

    plot_results(EXTRA_EPOCHS, train_losses, train_accs, valid_losses, valid_accs)

    model_testing(model, test_iterator, criterion, device, model_name=fine_tuned_model_path)
    print_report(model, test_iterator, device)

if __name__ == "__main__":
    main()