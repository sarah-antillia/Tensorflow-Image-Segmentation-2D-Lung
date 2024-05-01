<h2>Tensorflow-Image-Segmentation-2D-Lung (2024/05/01)</h2>

This is an experimental Image Segmentation project for 2D-Lung based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1cOSiTXeU_l8duN_DNTyPFnfeZEuMKodn/view?usp=sharing">
2D-Lung-ImageMask-Dataset-V1.zip</a> 
<br>
 
<hr>
Actual Image Segmentation Sample for an image.<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask_merged</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test/images/ID_0177_Z_0163.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test/masks/ID_0177_Z_0163.jpg" width="320" height="auto"></td>
<!--
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output/TCGA-CN-6019-01Z-00-DX1.0a01c44a-b7f4-427e-9854-fe2f0db631ab_1.jpg" width="320" height="auto"></td>
 -->
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output_merged/ID_0177_Z_0163.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this 2D Lung Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been taken from the following web-site<br>
<b>ORCA: ORal Cancer Annotated dataset</b><br>
<pre>https://sites.google.com/unibas.it/orca/home</pre>

<pre>
If you use the ORCA data, please cite:
F.  Martino,  D.D.  Bloisi,  A.  Pennisi,  M. Fawakherji,  G. Ilardi,  D. Russo,  D. Nardi,  S. Staibano, F. Merolla
"Deep Learning-based Pixel-wise Lesion Segmentation on Oral Squamous Cell Carcinoma Images"
Applied Sciences: 2020, 10(22), 8285; https://doi.org/10.3390/app10228285  [PDF]
</pre>
<br>

<br>

<h3>
<a id="2">
2 2D Lung ImageMask Dataset
</a>
</h3>
 If you would like to train this 2D-Lung Segmentation model by yourself,
 please download the latest normalized dataset from the google drive 
<a href="https://drive.google.com/file/d/1cOSiTXeU_l8duN_DNTyPFnfeZEuMKodn/view?usp=sharing">
2D-Lung-ImageMask-Dataset-V1.zip</a>.<br>


<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be
<pre>
./dataset
└─2D-Lung
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>


<b>2D Lung Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/2D-Lung-ImageMask-Dataset-V1_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid dataset is not necessarily large. 
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
4 Train TensorflowUNet Model
</h3>
 We have trained 2D-Lung TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/2D-Lung and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<pre>
; train_eval_infer.config
; 2024/05/01 (C) antillia.com

[model]
model         = "TensorflowUNet"
generator     = False
image_width    = 640
image_height   = 640
image_channels = 3
input_normalize = False
num_classes    = 1
base_filters   = 16
base_kernels   = (5,5)
num_layers     = 7
dropout_rate   = 0.08
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (2,2)
loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 2
patience      = 10
;metrics       = ["iou_coef", "val_iou_coef"]
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/2D-LUNG/train/images/"
mask_datapath  = "../../../dataset/2D-LUNG/train/masks/"
create_backup  = False
learning_rate_reducer = True
reducer_patience      = 5
save_weights_only = True

[eval]
image_datapath = "../../../dataset/2D-LUNG/valid/images/"
mask_datapath  = "../../../dataset/2D-LUNG/valid/masks/"

[test] 
image_datapath = "../../../dataset/2D-LUNG/test/images/"
mask_datapath  = "../../../dataset/2D-LUNG/test/masks/"

[infer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output"
merged_dir   = "./mini_test_output_merged"
;binarize      = True
;sharpening   = True

[segmentation]
colorize      = False
black         = "black"
white         = "green"
blursize      = None

[mask]
blur      = False
blur_size = (3,3)
</pre>

The training process has just been stopped at epoch 97 by an early-stopping callback as shown below.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/train_console_output_at_epoch_61.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
5 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/2D-Lung</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for 2D-Lung.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/evaluate_console_output_at_epoch_61.png" width="720" height="auto">
<br><br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) score for this test dataset is very low as shown below.<br>
<pre>
loss,0.0268
binary_accuracy,0.9909
</pre>

<h3>
6 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/2D-Lung</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for 2D-Lung.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
mini_test_images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/mini_test_images.png" width="1024" height="auto"><br>
mini_test_mask(ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>


<table>
<tr>
<th>Image</th>
<th>Inferred-mask</th>
<th>Inferred_mask_merged</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test/images/ID_0013_Z_0047.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output/ID_0013_Z_0047.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output_merged/ID_0013_Z_0047.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test/images/ID_0028_Z_0125.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output/ID_0028_Z_0125.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output_merged/ID_0028_Z_0125.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test/images/ID_0073_Z_0076.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output/ID_0073_Z_0076.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output_merged/ID_0073_Z_0076.jpg" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test/images/ID_0177_Z_0163.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output/ID_0177_Z_0163.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/2D-Lung/mini_test_output_merged/ID_0177_Z_0163.jpg" width="320" height="auto"></td>
</tr>
</table>
<br>
<br>


<hr>

<h3>
References
</h3>
