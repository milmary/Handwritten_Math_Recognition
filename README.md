---
title: OCR Ltx
emoji: 👀
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
---

# Handwritten Mathematical Expression Recognition

A deep learning system for recognizing **handwritten mathematical expressions** and converting them into LaTeX.

The project was developed as a bachelor's thesis and compares two neural network approaches to handwritten mathematical expression recognition: a **CRNN with CTC decoding** and a **CRNN encoder with an attention-based decoder**.

The application provides an interactive Gradio interface for uploading or capturing handwritten expressions, recognizing them, rendering the resulting LaTeX, and optionally solving recognized equations.

## Features

* Handwritten mathematical expression recognition
* Image-to-LaTeX conversion
* Two neural network architectures:

  * CRNN + CTC
  * CRNN Encoder + Attention Decoder
* Interactive model selection
* LaTeX rendering and live preview
* Editable recognition output
* Symbolic equation solving using SymPy
* Image upload and webcam input
* Automatic CPU, CUDA, and Apple MPS support
* Interactive Gradio web interface

## Architecture

### CRNN + CTC

The first model uses a convolutional neural network to extract visual features from an input image, followed by a bidirectional LSTM for sequence modelling.

The output sequence is decoded using **Connectionist Temporal Classification (CTC)**, allowing the model to recognize expressions without explicit character-level segmentation.

```text
Image
  ↓
CNN Feature Extractor
  ↓
Bidirectional LSTM
  ↓
Linear Classifier
  ↓
CTC Decoding
  ↓
LaTeX
```

### CRNN + Attention

The second model follows an encoder-decoder architecture.

A CNN and bidirectional LSTM encode the handwritten expression into a sequence of visual features. An attention-based recurrent decoder then generates the LaTeX representation token by token.

```text
Image
  ↓
CNN Feature Extractor
  ↓
Bidirectional LSTM Encoder
  ↓
Attention
  ↓
LSTM Decoder
  ↓
LaTeX
```

## Application

The Gradio interface allows the user to:

1. Upload an image or capture one using a webcam.
2. Select either the CTC or Attention model.
3. Recognize the handwritten expression.
4. View the generated LaTeX and rendered mathematical expression.
5. Edit the recognized LaTeX if necessary.
6. Solve supported expressions symbolically.

Symbolic processing and equation solving are implemented using **SymPy**.

## Technologies

* Python
* PyTorch
* Torchvision
* Gradio
* SymPy
* Pillow
* Deep Learning
* CNN / CRNN
* Bidirectional LSTM
* CTC
* Attention-based Encoder–Decoder

## Project Structure

```text
Handwritten_Math_Recognition/
├── models/              # Trained model checkpoints
├── app.py               # Models, inference and Gradio application
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/milmary/Handwritten_Math_Recognition.git
cd Handwritten_Math_Recognition
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

The application automatically uses CUDA or Apple MPS when available and otherwise falls back to CPU.

## Model Comparison

Two approaches to handwritten mathematical expression recognition were implemented and evaluated:

| Model            | Decoding           | Characteristics                                        |
| ---------------- | ------------------ | ------------------------------------------------------ |
| CRNN + CTC       | Non-autoregressive | Simpler and faster sequence decoding                   |
| CRNN + Attention | Autoregressive     | Generates LaTeX token by token using learned attention |

The comparison investigates the trade-offs between CTC-based sequence recognition and attention-based encoder-decoder modelling for handwritten mathematical expressions.

## Dataset

The models were trained and evaluated on the **HME100K handwritten mathematical expression dataset**.
