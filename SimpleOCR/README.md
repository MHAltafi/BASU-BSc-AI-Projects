# Persian OCR Using PyTorch

A deep learning project for recognizing printed Persian text from images, developed during my summer internship in 2024.

This project explores an end-to-end Optical Character Recognition (OCR) pipeline using a custom **CNN–BiLSTM architecture with Connectionist Temporal Classification (CTC) loss**. A synthetic dataset was created from Iran's province names rendered in 39 Persian fonts. The images were then manually divided into training, validation, and test sets by fonts.

The project covers synthetic dataset generation, image preprocessing, deep learning model development, training experiments, and OCR evaluation.

## Features

- Synthetic Persian text image generation using 39 Persian fonts.
- Persian text shaping and right-to-left rendering.
- Image cropping to remove excess whitespace.
- Grayscale image preprocessing and resizing.
- Custom CNN with residual convolutional blocks.
- Bidirectional LSTM for sequence modeling.
- CTC loss for sequence-based text recognition.
- Character Error Rate (CER) and Word Error Rate (WER) evaluation.
- Training experiments, model checkpoints, and result tracking.
- Prediction examples for qualitative evaluation.

## Project Structure

Only the two Jupyter notebooks are tracked in this repository. All other files and directories (fonts, generated images, dataset splits, trained models) are excluded via `.gitignore`. To test the code, you need to provide the font files yourself: download some Persian `.ttf` / `.otf` fonts and place them in a `Fonts/` folder inside `SimpleOCR/`, then run `create_dataset.ipynb` to generate the synthetic images (written to `output/` and `output2/`). Next, manually divide the generated images into three folders named `train/`, `valid/`, and `test/` inside `dataset/`, and finally run `OCR.ipynb` to train and evaluate the model.

```text
SimpleOCR/
├── .gitignore
├── create_dataset.ipynb
└── OCR.ipynb
```

| File | Description |
|---|---|
| `create_dataset.ipynb` | Generates synthetic Persian text images from fonts and crops/normalizes them. |
| `OCR.ipynb` | Contains the dataset loader, model architecture, training, and evaluation code. |

### Artifacts not tracked in this repository

Running the notebooks locally will recreate the following directories and files. They are intentionally not committed to keep the repository small and free of generated data.

| Path | Description |
|---|---|
| `Fonts/` | Persian `.ttf` / `.otf` font files used for image generation. |
| `dataset/` | Training, validation, and test images, split manually. |
| `output/` | Raw generated text images. |
| `output2/` | Cropped and processed images. |
| `models/` | Saved model checkpoints. |
| `models_results.csv` | Recorded model experiment results. |
| `ocr_data.csv` | Recorded OCR-related experimental data. |

## Dataset

### Synthetic dataset generation

The dataset was created by rendering the names of Iran's 31 provinces using 39 Persian fonts.

The text images were generated using Pillow, with Persian text shaping and bidirectional text handling provided by `arabic-reshaper` and `python-bidi`.

The generated images use black text on a white background. The initial images are saved in `output/`, and cropped versions are saved in `output2/`.

### Preprocessing

The image-processing pipeline includes:

1. Loading available `.ttf` and `.otf` font files.
2. Rendering Persian province names into images.
3. Handling Persian character shaping and text direction.
4. Cropping excess whitespace around the rendered text.
5. Converting images to grayscale and resizing them for model input.
6. Normalizing image pixel values.
7. Applying random rotation during training as data augmentation.

For model training, the input images are resized to **32 × 128 pixels**.

### Dataset splitting

The processed images were manually divided into three subsets:

- **Training set (`dataset/train/`):** Used to train the model.
- **Validation set (`dataset/valid/`):** Used to monitor performance and compare experiments.
- **Test set (`dataset/test/`):** Intended for final model evaluation.

The split was performed manually rather than by an automated splitting script.

The dataset consists of synthetic, computer-rendered text. It does not fully represent the challenges of real-world OCR, such as handwriting, noisy scans, perspective distortion, or complex backgrounds.

## Model Architecture

The OCR model combines a convolutional feature extractor with a bidirectional recurrent network.

### Architecture overview

1. **Input:** A grayscale image resized to 32 × 128 pixels.
2. **CNN feature extractor:** Five residual convolutional blocks extract visual features.
3. **Sequence conversion:** The spatial feature map is reshaped into a sequence of feature vectors.
4. **BiLSTM:** A two-layer bidirectional Long Short-Term Memory network models contextual relationships across the sequence.
5. **Output layer:** A linear layer produces character-class scores for each time step.
6. **CTC loss and decoding:** CTC handles alignment between the predicted sequence and the target text without requiring character-level image segmentation.

The network uses convolutional layers, batch normalization, residual connections, ReLU activations, and dropout.

### Training configuration

| Parameter | Configuration |
|---|---|
| Deep learning framework | PyTorch |
| Input image size | 32 × 128 |
| Input channels | 1 (grayscale) |
| CNN feature extractor | Five residual convolutional blocks |
| Recurrent network | Two-layer bidirectional LSTM |
| Hidden units per direction | 128 |
| Optimizer | Adam |
| Loss function | CTC Loss |
| Initial learning rate | 0.01 |
| Other explored learning rate | 0.001 |
| Dropout | 0.5 in the main configuration |
| Random seed | 1 |
| Training batch size | 8 |

These values summarize the documented notebook configuration. Individual experiments may use different settings.
## Installation

### Requirements

The project uses Python and the following libraries. The versions below were captured from the development environment and are provided as a reference; other versions may also work.

| Package | Version |
|---|---|
| PyTorch | 2.2.1 (CUDA 11.8) |
| torchvision | 0.17.1 (CUDA 11.8) |
| Pillow | 10.3.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| Matplotlib | 3.8.4 |
| Seaborn | 0.13.2 |
| tqdm | 4.66.4 |
| arabic-reshaper | 3.0.0 |
| python-bidi | 0.6.0 |
| torchmetrics | 1.9.0 |
| torcheval | 0.0.7 |
| Jupyter Notebook | (any recent version) |

> **Note:** PyTorch and torchvision were installed with CUDA 11.8 builds. If you don't have a CUDA 11.8–compatible GPU, install the CPU builds instead — the code still runs on CPU, just more slowly.

### Setup

This project lives inside a monorepo. You can either download only this folder, or clone the whole repository.


After downloading this folder, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you have a CUDA 11.8–compatible GPU and want the GPU builds of PyTorch and torchvision instead of the default CPU builds, install them first:

```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining requirements:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `create_dataset.ipynb` to inspect the dataset-generation pipeline, or open `OCR.ipynb` to explore the model, training, and evaluation code.

A CUDA-compatible GPU is recommended for faster training. The notebook's device configuration determines whether CUDA or the CPU is used.

## Usage

### 1. Prepare the fonts

Create a `Fonts/` directory inside `SimpleOCR/` and place Persian `.ttf` / `.otf` font files there. The notebooks read fonts from this directory. The font files are not included in this repository, so you need to download them yourself.

### 2. Generate synthetic images

Open `create_dataset.ipynb` and execute the cells in order.

The notebook generates raw images into `output/` and writes cropped/processed versions into `output2/`. These output directories are created automatically if missing. Their contents are ignored by `.gitignore`.

### 3. Prepare the dataset

After generating the images, manually divide them into three folders inside `dataset/`:

```text
dataset/
├── train/
├── valid/
└── test/
```

The original dataset was split manually. If you reproduce the split, keep the training, validation, and test sets separate.

The dataset loader derives text labels from image filenames. Therefore, filenames must follow the naming convention expected by `OCR.ipynb`.

### 4. Train the OCR model

Open `OCR.ipynb` and execute the cells in order.

The notebook includes:

- Dataset loading and image transformations.
- Character-to-index encoding.
- CNN–BiLSTM model initialization.
- Training and validation.
- Learning-rate experiments.
- Checkpoint saving.
- Loss and OCR metric tracking.
- Visualization of predictions.

Check the dataset paths and model-saving paths before starting a new training run. Saved checkpoints go to `models/`, which is ignored by `.gitignore`.

### 5. Evaluate predictions

The notebook uses two metrics:

- **Character Error Rate (CER):** Measures character-level edit errors relative to the reference text.
- **Word Error Rate (WER):** Measures word-level edit errors relative to the reference text.

The notebook also compares predicted text with ground-truth labels to help identify recognition errors.

For reliable evaluation, calculate both metrics over the complete test set rather than a single batch.

## Results

The recorded experiments include a model configuration with the following reported values:

| Metric | Recorded value |
|---|---:|
| Character Error Rate (CER) | 0.0797 |
| Word Error Rate (WER) | 0.1599 |

These correspond to approximately 7.97% CER and 15.99% WER.

Lower values indicate better recognition performance. The notebook also contains examples of correct predictions and errors such as character substitutions or omissions.

**Evaluation note:** These values summarize recorded experimental results; they should not be treated as independently verified, full-test-set benchmark scores.

The results also depend on the manually created dataset split and the distribution of fonts across the subsets. Performance on synthetic images does not necessarily indicate equivalent performance on real-world Persian documents.

## Limitations and Future Improvements

This project was developed as an experimental OCR implementation during a summer internship in 2024.

Potential improvements include:

- **Real-world OCR:** Introduce scanned documents, image noise, and varied backgrounds.
- **Expanded vocabulary:** Extend recognition beyond the names of the 31 Iranian provinces.
- **Inference interface:** Add a simple interface for recognizing text in a new image.

## Acknowledgments

This project was developed as part of my summer internship in 2024. It provided practical experience with synthetic dataset creation, image preprocessing, CNN–BiLSTM model design, sequence recognition, and OCR evaluation.