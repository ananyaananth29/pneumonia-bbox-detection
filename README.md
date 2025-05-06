# Pneumonia Detection from Chest X-rays using RetinaNet

This project implements an object detection model to identify pneumonia-related lung opacities in chest X-ray images using the RSNA Pneumonia Detection Challenge dataset. It leverages a RetinaNet model with a ResNet-101 backbone and Feature Pyramid Network for multi-scale bounding box prediction.

##  Dataset

* **Source**: [RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)
* **Format**: DICOM images with corresponding bounding box annotations in CSV format.
* **Train/Val/Test**: 80/20 split used for evaluation. A subset of the official test set was used for qualitative analysis.

##  Model Architecture

* **Backbone**: ResNet-101 (pretrained on ImageNet)
* **Detection Framework**: RetinaNet with FPN
* **Anchors**: Multi-scale with aspect ratios of (0.5, 1.0, 2.0)
* **Losses**:

  * Classification: Focal Loss
  * Regression: Smooth L1 Loss
* **Optimizer**: AdamW
* **Learning Rate Schedule**: StepLR (decay every 10 epochs)
* **Early Stopping**: Patience of 7 epochs

##  Training

* Images resized to **768x768**
* Normalized and converted to single-channel tensors
* Early stopping based on validation loss
* Augmentations were explored but not used in final training

##  Evaluation Metrics

Evaluation was done using the `torchmetrics` Mean Average Precision (mAP) metric.

| Metric      | Score  |
| ----------- | ------ |
| mAP\@0.5    | 0.6205 |
| Overall mAP | 0.6205 |
| Recall\@100 | 0.7614 |

##  Files

* `MyProject.py`: Main training and evaluation script
* `test.ipynb`: Notebook to test model predictions and visualize bounding boxes
* `saved_model.pth`: Trained RetinaNet model
* `Test_predictions/`: Folder containing predicted X-rays with bounding boxes

##  Visualization

Green boxes on the test X-rays indicate predicted regions of pneumonia. Confidence scores are shown for each detection. See examples:

![image](https://github.com/user-attachments/assets/be0e5d48-363b-4ddc-9ed2-27f33a657c26)

##  Future Work

* Explore transformer-based models (e.g., DETR)
* Use test-time augmentation
* Fine-tune on external datasets
* Apply semi-supervised learning to utilize unlabeled data

##  Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/ananyaananth29/pneumonia-bbox-detection.git
   cd pneumonia-bbox-detection
   ```

2. Link to Dataset:


    [RSNA Pneumonia Detection Challenge (Kaggle)](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge)


3. Download dataset and place under `rsna_data/` with structure:

   ```
   rsna_data/
     ├── stage_2_train_images/
     ├── stage_2_test_images/
     └── stage_2_train_labels.csv
   ```

4. Run training:

   ```bash
   python MyProject.py
   ```

5. Run predictions (optional):

   ```bash
   jupyter notebook test.ipynb
   ```

##  Acknowledgments

This project was completed as part of the Deep Learning course final project by:

* Ananya Ananth (u1520797)
* Manjusha Muppala (u1528137)
