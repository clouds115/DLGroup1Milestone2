# deepLearningUPMGroup01

Deep Learning Course 2025/26 - Group 1: Alonso Geesink Anton, Claudio Vincenzo Catalano Leiva, Seyit Ahmet, Inci

## Fake News Detection with DistilBERT

This repository contains a Jupyter notebook for fine-tuning a DistilBERT model to detect fake news articles.

### Dataset

The project uses the [Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data) from Kaggle, which contains:
- **Fake.csv**: Fake news articles
- **True.csv**: True news articles

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/clouds115/deepLearningUPMGroup01.git
cd deepLearningUPMGroup01
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- jupyter

#### 3. Download the Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)
2. Create a `data/` directory in the project root
3. Place `Fake.csv` and `True.csv` in the `data/` directory

```bash
mkdir -p data
# Download Fake.csv and True.csv to the data/ directory
```

#### 4. Run the Notebook

```bash
jupyter notebook fake_news_detection_distilbert.ipynb
```

### Project Structure

```
deepLearningUPMGroup01/
├── fake_news_detection_distilbert.ipynb  # Main training notebook
├── example_inference.py                  # Example inference script
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── .gitignore                           # Git ignore rules
├── data/                                # Dataset directory (not tracked)
│   ├── Fake.csv
│   └── True.csv
└── models/                              # Saved models (not tracked)
    └── fake_news_distilbert/
```

### Notebook Overview

The `fake_news_detection_distilbert.ipynb` notebook includes:

1. **Setup and Imports**: Load required libraries
2. **Data Loading**: Load and combine fake/true news datasets
3. **Data Exploration**: Visualize dataset distribution
4. **Preprocessing**: Clean and split data into train/val/test sets
5. **Tokenization**: Prepare data using DistilBERT tokenizer
6. **Model Setup**: Configure DistilBERT for binary classification
7. **Training**: Fine-tune the model with progress tracking
8. **Visualization**: Plot training curves and metrics
9. **Evaluation**: Comprehensive metrics on test set
10. **Inference**: Predict on new articles
11. **Model Saving**: Save trained model and tokenizer

### Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary sequence classification (Fake vs True)
- **Parameters**: ~66M trainable parameters
- **Max Sequence Length**: 512 tokens

### Training Configuration

Default hyperparameters:
- **Batch Size**: 16
- **Epochs**: 3
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Scheduler**: Linear with warmup

### Expected Results

The model typically achieves:
- **Accuracy**: 98-99% on test set
- **Precision/Recall**: High performance on both classes
- **Training Time**: ~30-60 minutes (depending on hardware)

### Usage Example

After training, use the model for inference:

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('models/fake_news_distilbert')
tokenizer = DistilBertTokenizer.from_pretrained('models/fake_news_distilbert')

# Predict
text = "Your news article text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print("Fake" if prediction == 0 else "True")
```

**Or use the provided example script:**

```bash
python example_inference.py
```

This script provides:
- Pre-defined example predictions
- Interactive mode for testing custom text
- Confidence scores for each prediction

### References

This implementation is based on the following Kaggle notebooks:
- [Fake and Real News DistilBERT](https://www.kaggle.com/code/henryukwuoma/fake-and-real-news-distilbert)
- [DistilBERT Fine-tuning on Fake News Data](https://www.kaggle.com/code/shehabmagdy710/distilbert-fine-tuning-on-fake-news-data)

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slower training)
- **Recommended**: GPU with 8GB+ VRAM (CUDA-compatible)
- **Disk Space**: ~5GB (model, data, checkpoints)

### Troubleshooting

**Out of Memory Error:**
- Reduce `BATCH_SIZE` to 8 or 4
- Reduce `MAX_LENGTH` to 256

**Slow Training:**
- Use GPU if available
- Reduce dataset size for testing
- Use fewer epochs

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Update to latest versions if needed

### License

This project is for educational purposes as part of the Deep Learning Course 2025/26 at UPM.

### Contributors

- Alonso Geesink Anton
- Claudio Vincenzo Catalano Leiva
- Seyit Ahmet
- Inci

### Contact

For questions or issues, please open an issue in the GitHub repository.
