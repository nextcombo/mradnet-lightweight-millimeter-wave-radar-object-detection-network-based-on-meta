# mRadNet: Lightweight Millimeter Wave Radar Object Detection Network Based on MetaFormer [with Python code]

mRadNet: A Lightweight Millimeter Wave Radar Object Detection Network Based on MetaFormer
A summary of the mRadNet: The introduction of the MetaFormer architecture into the field of millimeter wave radar object detection, achieving a perfect balance between high accuracy and lightweight through the clever integration of separable convolution (SepConv) and self-attention mechanism (Self-Attention) in the CRUW ROD2021 dataset.
I. Introduction: Why is mRadNet needed? In autonomous driving perception systems, millimeter wave radars have become an indispensable sensor due to their all-weather capabilities, low cost, and immunity to light. However, compared to lidar and cameras, radar data has characteristics such as high sparsity, low resolution, and poor signal-to-noise ratio, posing significant challenges to target detection.
In recent years, deep learning-based object detection methods have emerged continuously, such as T-RODNet, E-RODNet, etc. However, these methods generally face a problem: large model parameters and slow inference speed, making it difficult to meet the real-time deployment needs of edge devices in vehicles.
The mRadNet was born in this context. Its core idea is to leverage the lightweight general architecture paradigm of MetaFormer, combine the advantages of convolution and attention mechanisms, and build a compact and efficient radar detector.
Millimeter wave radars obtain Range-Angle (RA) heat maps through the transmission and reception of FMCW (Frequency Modulated Continuous Wave) signals. Each frame RA heat map is a (128×128) two-dimensional matrix, with each position containing complex numbers (real part + imaginary part). Each frame of radar will transmit multiple chirps (this work uses 4), so the input data dimension is:
[\text{Input} \in \mathbb{R}^{T \times C_h \times R \times A \times 2}]
Where:
(T) Time window length (16 frames)  
(C_h) Chirp count (4 chirps)  
(R), (A): Represent resolution for distance and angle (128×128)  
(2): Real part + imaginary part  
The target detection is divided into three categories: pedestrians (pedestrian), cyclists (cyclist), and cars (car). The model outputs a three-dimensional confidence heatmap (Confidence Map), with shape (R×A×3), where the numerical value at each position represents the existence of a certain type of target in that three-dimensional space and its confidence level.  
The evaluation metric is OLS (Object Location Similarity), which measures the similarity between the predicted target and the ground truth target in the polar coordinate system, mainly looking at two indicators: AP (Average Precision) and AR (Average Recall Rate).  
III. Details of mRadNet Network Architecture mRadNet adopts the classic **Encoder-Decoder** structure, as shown in the following figure:  
Next, we will dissect it layer by layer.  
3.1 Stem layer: Chirp merging and dimensionality increase input data first goes through a Stem layer, which is essentially a 3D convolution:  
self.stem = nn.Conv3d(2, 2*num_chirps, (num_chirps, 1, 1), stride=(num_chirps, 1, 1))\nInput channels = 2 (real part+imaginary part)  
Output channels = 8 ((2×4) chirps)  
Kernel size = (4, 1, 1), stride = (4, 1, 1)
The operation here merges the real and imaginary parts of 4 chirps along the chirp dimension, increasing the number of channels from 2 to 8 while also compressing the chirp dimension. This design avoids complex chirp dimension modeling, allowing subsequent networks to process features only in the spatiotemporal dimension.
Output tensor shape: (B\times T\times H\times W\times 8)
3.2 Encoder: The four-stage progressive feature extraction encoder consists of four stages, adopting the core concept of MetaFormer: decoupling the Token Mixer from the MLP into a modular architecture.
The configuration for each stage is as follows:
Stage  
Depth  
Feature dimension  
Type of Token Mixer  
Stage 1  
2  
48  
SepConv  
Stage 2  
2  
96  
SepConv  
Stage 3  
2  
192  
Self-Attention  
Stage 4  
1  
384  
Self-Attention  
The design is very clever:
Using SepConv for shallow layers and Attention for deeper ones, because:
Shallow layers capture high-resolution feature maps ((64\x64)) with efficient convolutional computation using separable convolutions, and are good at capturing local spatial features;
Deeper layers use self-attention mechanisms due to their low resolution ((8\x8)) and fewer tokens, making the computational overhead controllable for self-attention and effectively modeling long-range dependencies.
Token Embedding (Patch embedding) input first goes through a TokenEmbed layer, where a 3D convolution is used to cut the original RA graph into patches and map them to the embedding space:
Patch Size = (2, 2, 2), meaning downsampling by a factor of 2 in time, distance, and angle dimensions.
Convolutional Kernel = Step Size = (2, 2, 2), with an output channel of dims[0] = 48.
Separated Convolution (SepConv) borrowed from MobileNetV2:
Input → Linear(dim → 2×dim) → StarReLU → DWConv3D(Depthwise) → Identity → Linear(2×dim → dim).
The key feature is to first expand the dimension (expansion ratio=2) before performing depthwise separable convolutions, then reduce back to the original dimension—reducing parameter count while maintaining expressiveness.
Attention Mechanism (Attention) standardized Scaled Dot-Product Attention but adapted for 5D tensors (B, T, H, W, C):
Perform attention computation on all spatial tokens ((T×H×W) sized) together.
Utilize PyTorch's scaled_dot_product_attention to implement efficient Flash Attention.
Token Merging Downsample adjacent stages using TokenMerging for 2x spatial downsampling: concatenate adjacent 2×2 spatial regions through a linear layer to reduce dimensionality:
[B, T, H, W, C] → split into 4 offsets → concat → [B, T, H/2, W/2, 4C] → Linear → [B, T, H/2, W/2, 2C].
This spatial-to-channel transformation retains more spatial information than simple pooling or stride convolution downsampling.
3.3 Decoder: The decoder is the "mirror" of the encoder, also based on the MetaFormer module but with three key adjustments:
- Token Mixer strategy inversion stage
- Deepness
- Token Mixer
- Stage 1
2
- Attention
- Stage 2
2
- Separable Convolution
- Stage 3
2
SepConv
In contrast to the encoder: deep (small size) first uses Attention to fuse global semantics, and shallow (large size) uses Separable Convolution to refine local details.
Upsampling and skip connections are performed at each stage through ConvTranspose3d for a 2x spatial upsampling, and are connected to the feature maps of corresponding layers in the encoder (Skip Connection, concatenation along channel dimension):
x → Linear → ConvTranspose3d(upsample) + Skip(encoder feature) → MetaFormer Stage → ...
Finally, after the last stage, trilinear upsampling is used to restore the feature maps to the original RA resolution, and a 3×3 convolution is applied to obtain the final output heatmap:
[\text{Output} \in \mathbb{R}^{B\times T\times R\times A\times 3}]
3.4 The overall design of the model mRadNet aims for compactness:
Encoder dims: [48, 96, 192, 384]
Decoder embed_dim: 64
Output categories: 3 (pedestrian/cyclist/car)
4. Training Strategy and Data Processing
4.1 Dataset Usage: The radar part of the CRUW ROD2021 dataset is used for training:
- Training set: 36 sequences
- Test set: 4 sequences
Each sequence contains tens of thousands of RA heatmap frames
4.2 Sliding Window Sampling: A time sliding window strategy organizes the input:
- Window size: 16 frames
- Step size during training: 4 frames (overlapping sampling, increasing data volume)
- Step size during testing: 8 frames
4.3 Data Augmentation: Two types of augmentation are triggered with a 50% probability each:
1. Time flipping: Reverse the order of frames within the window
2. Horizontal flipping: Invert the angle dimension (left-right symmetry)
These enhancements fully utilize the spatiotemporal symmetry of radar data, effectively preventing overfitting.
4.4 Loss Function: Use Smooth L1 Loss to directly optimize confidence heatmaps:
loss_fn = nn.SmoothL1Loss()
loss = loss_fn(outputs['output'], confmap_gt)
Compared to MSE, Smooth L1 is more robust against outliers, suitable for radar data where noise peaks may occur.
4.5 Optimization Configuration Parameters:
Configuration
Optimizer
AdamP
Initial learning rate
(1×10^-4)
Learning rate scheduling
Cosine Annealing
Training rounds
20
Batch Size
4
5.1 Model Foreground Inference Testing: When batch_size=1, process frame by frame during testing. After the model outputs a confidence heatmap, it needs to be decoded into discrete target detection results.
5.2 Confidence Heatmap Decoding (Confmap Decoding): The decoding process is divided into two steps:
Step 1: Detecting local peak values in a 3x3 neighborhood
For each channel (category), a confidence heatmap is generated using the `find_3x3_peaks` function to locate all local maxima within 3×3 neighborhoods—i.e., rolling comparison of up, down, left, right, and four diagonal directions.
Step 2: L-NMS (Position Non-Maximum Suppression)
The NMS filtering is performed using the OLS (Object Location Similarity) metric designed for radar data:
[OLS(p_1, p_2) = exp(-dx2 + dy2/(2*s^2*κ))], where (s) is the radial distance to the target and (\kappa) is the category-related scale factor. This design differs from traditional IoU-NMS and is more suitable for radar detection scenarios in polar coordinates.
Finally, the results are sorted by confidence and top-100 detections are retained.
Evaluation metrics are based on the ROD2021 official evaluation protocol, using 9 OLS thresholds with intervals of 0.5~0.9 and a 0.05 interval. The AP (Average Precision) and AR (Average Recall) are calculated.
Method highlights:
- MetaFormer paradigm modularizes the Token Mixer, facilitating flexible replacement of different mixer types and unifying convolution and attention architectures.
- The mixed Token Mixer strategy employs SepConv for shallow layers to efficiently extract local features and Self-Attention for deep layers to model long-range dependencies—"Convolution sees local, Attention sees global."
- The Chirp processing only uses a single 3D convolution to merge chirp dimensions at the Stem layer, avoiding complex sequential modeling modules and significantly reducing computational load.
🔹 Lightweight design through techniques such as separable convolution and Token Merging downsampling, allows the model to maintain high performance while having lower parameters and computational requirements, suitable for edge deployment in vehicles.

Step 1: Download data
Download TRAIN_RAD_H.zip and TRAIN_RAD_H_ANNO.zip from the CRUW official website, and place them according to the following structure:
├─ annotations/│ ├─ test/ (4 sequences)│ └─ train/ (36 sequences)└─ sequences/ ├─ test/ (4 sequences) └─ train/ (36 sequences)

Step 2: Configure environment
conda create -n mradnet python=3.12 conda activate mradnet pip install -r requirements.txt

Step 3: Training and Testing
# Modify the data path in config/mRadNet.yaml for training with python train.py -c config/mRadNet.yaml # Train with python test.py -c config/mRadNet.yaml -r checkpoint.pt # Test By rationally allocating convolution and attention mechanisms in "layer roles," it achieves precise detection of targets in radar RA heat maps without adding excessive parameters.
This design philosophy—"a unified large framework, differentiated small designs"—might be an important direction for future lightweight perception models. Whether it's in automotive radar, security radar, or smart home scenarios, mRadNet provides a valuable paradigm.  
Paper: mRadNet: A Compact Radar Object Detector with MetaFormer

## Images

![mRadNet: Lightweight Millimeter Wave Radar Object Detection Network Based on MetaFormer [with Python code]](cover.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
