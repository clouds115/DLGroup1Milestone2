# UPM EMSE Deep Learning 2025/26 Course - Course Project

Group 1: Alonso Geesink Anton, Claudio Vincenzo Catalano Leiva, Seyit Ahmet Inci

## Fake News Detection with DistilBERT

### Dataset

Our project will be trained using the [Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data) from Kaggle, which contains:
- **Fake.csv**: Fake news articles
- **True.csv**: True news articles

### Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/clouds115/deepLearningUPMGroup01.git
cd DLGroup1Milestone2
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

### Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary sequence classification (Fake vs True)
- **Parameters**: ~66M trainable parameters
- **Max Sequence Length**: 128 tokens
- **Weight Decay**: 0.02

### Training Configurations

## C1
- **Batch Size**: 32
- **Epochs**: 1
- **Learning Rate**: 3e-5

## C2
- **Batch Size**: 32
- **Epochs**: 1
- **Learning Rate**: 1.5e-5

## C3
- **Batch Size**: 32
- **Epochs**: 2
- **Learning Rate**: 3e-5

## C4
- **Batch Size**: 32
- **Epochs**: 2
- **Learning Rate**: 1.5e-5

## C5
- **Batch Size**: 32
- **Epochs**: 3
- **Learning Rate**: 3e-5

## C6
- **Batch Size**: 32
- **Epochs**: 3
- **Learning Rate**: 1.5e-5
- 
### UI Implementation - Milestone 3

To view the work performed during milestone 3 for the UI implementation, please refer to the following repository: https://github.com/alonso113/DLGroup1Milestone3

### References

This implementation is based on the following Kaggle notebooks as template code:
- [Fake and Real News DistilBERT](https://www.kaggle.com/code/henryukwuoma/fake-and-real-news-distilbert)
- [DistilBERT Fine-tuning on Fake News Data](https://www.kaggle.com/code/shehabmagdy710/distilbert-fine-tuning-on-fake-news-data)

### License

Educational project - Apache 2.0 (DistilBERT model)

### Contributors

- Alonso Geesink Anton
- Claudio Vincenzo Catalano Leiva
- Seyit Ahmet Inci
