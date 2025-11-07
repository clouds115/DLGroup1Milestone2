"""
Example script for using the trained DistilBERT fake news detection model.

This script demonstrates how to:
1. Load a trained model and tokenizer
2. Make predictions on new text
3. Get confidence scores

Usage:
    python example_inference.py
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


def predict_news(text, model, tokenizer, device, max_length=512):
    """
    Predict whether a news article is fake or true.
    
    Args:
        text: News article text
        model: Trained model
        tokenizer: Tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length
    
    Returns:
        prediction: 'Fake' or 'True'
        confidence: Confidence score (0-1)
    """
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
    prediction = 'True' if predicted_class.item() == 1 else 'Fake'
    
    return prediction, confidence.item()


def main():
    """Main function to demonstrate model inference."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model and tokenizer
    print('\nLoading model and tokenizer...')
    model_path = 'models/fake_news_distilbert'
    
    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = model.to(device)
        print('✓ Model loaded successfully!')
    except Exception as e:
        print(f'✗ Error loading model: {e}')
        print('\nPlease train the model first by running the notebook:')
        print('  fake_news_detection_distilbert.ipynb')
        return
    
    # Example news articles
    examples = [
        {
            'title': 'Scientific Discovery',
            'text': """
            Scientists have discovered a new species of dinosaur in Argentina. 
            The fossil, believed to be 95 million years old, was found by paleontologists 
            working in the Patagonia region. The discovery provides new insights into 
            the evolution of carnivorous dinosaurs.
            """
        },
        {
            'title': 'Shocking News',
            'text': """
            BREAKING: World leaders shocked by this one simple trick! 
            You won't believe what happens next! Scientists hate him!
            Click here to find out the secret they don't want you to know!
            """
        },
        {
            'title': 'Economic Report',
            'text': """
            The Federal Reserve announced today that interest rates will remain unchanged
            for the third consecutive meeting. The decision comes amid concerns about
            inflation and economic growth. Market analysts predict this will have
            significant implications for mortgage rates and consumer spending.
            """
        }
    ]
    
    print('\n' + '='*80)
    print('PREDICTIONS')
    print('='*80)
    
    for i, example in enumerate(examples, 1):
        text = f"{example['title']} {example['text']}"
        prediction, confidence = predict_news(text, model, tokenizer, device)
        
        print(f'\n📰 Example {i}: {example["title"]}')
        print(f'Text preview: {example["text"][:150].strip()}...')
        print(f'Prediction: {prediction}')
        print(f'Confidence: {confidence:.2%}')
        print('-'*80)
    
    # Interactive mode
    print('\n' + '='*80)
    print('INTERACTIVE MODE')
    print('='*80)
    print('Enter your own news article text (or "quit" to exit):')
    
    while True:
        print('\n> ', end='')
        user_text = input().strip()
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            print('Goodbye!')
            break
        
        if not user_text:
            continue
        
        prediction, confidence = predict_news(user_text, model, tokenizer, device)
        print(f'\nPrediction: {prediction}')
        print(f'Confidence: {confidence:.2%}')


if __name__ == '__main__':
    main()
