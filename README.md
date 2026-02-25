# Deep Learning Multi-label Classification Project

This project implements a multi-label classification system to predict four attributes for a given set of images.

## Project Deliverables

1. **Training Code**: `train.py` handles model training, data loading, and weight saving.
2. **Loss Curve Plot**: `Aimonk_multilabel_problem.png` shows the training loss over iterations.
3. **Inference Code**: `inference.py` predicts attributes for a single image.
4. **Dataset Handling**: `dataset.py` parses labels and manages image loading.

## Techniques and Implementation Details

### 1. Handling Missing Data (NA Values)

The dataset contains "NA" values for some attributes, indicating missing information. To handle this without ignoring the entire image (as requested), I implemented a **Masked Binary Cross Entropy Loss**.

- **Transformation**: "NA" values are mapped to a special value (-1).
- **Masking**: During loss calculation, a binary mask is created: `mask = (labels >= 0)`.
- **Loss Computation**: The standard BCE loss is multiplied by this mask, ensuring that gradients only flow from valid (0 or 1) labels. The final loss is averaged only over the non-masked entries.

### 2. Addressing Class Imbalance

The dataset is imbalanced across attributes. To mitigate this:

- **Weighted Loss**: I implemented automatic calculation of **Positive Weights** (`pos_weight`) for each attribute. This scales the loss for the minority class (1s) relative to the majority class (0s), ensuring the model focuses on learning underrepresented features.
- **Pre-trained Model**: Using a pre-trained ResNet18 model provides a strong feature extractor, which helps the model generalize better despite imbalance.

### 3. Model Architecture

I used a pre-trained **ResNet18** model. ResNet is an established architecture known for its stability and performance due to skip connections. I replaced the final fully connected layer with a new linear layer with 4 output units to match our task.

### 4. Training Process

- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Preprocessing**: Images are resized to 224x224 and normalized using ImageNet statistics.
- **Monitoring**: Loss is tracked at every iteration and plotted at the end.

## Future Improvements (Given More Time)

1. **Data Augmentation**: Implementing techniques like random flip, rotation, and color jitter to increase model robustness.
2. **Weighted Random Sampler**: Using a sampler to balance the attributes in each batch.
3. **Hyperparameter Tuning**: Experimenting with different learning rates, batch sizes, and architectures (e.g., EfficientNet).
4. **Ensemble Modeling**: Combining predictions from multiple models.
5. **Validation Metrics**: Tracking F1-score or Precision-Recall curves, which are more informative for imbalanced multi-label tasks than simple loss.

## How to Run

### Training

```bash
python train.py
```

This will generate `model.pth` and `Aimonk_multilabel_problem.png`.

### Inference

```bash
python inference.py <image_path>
```

Example:

```bash
python inference.py images/image_0.jpg
```
