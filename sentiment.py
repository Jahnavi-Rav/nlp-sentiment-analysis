from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn

class SentimentAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        sentiment = ['negative', 'neutral', 'positive'][predictions.argmax().item()]
        return {'sentiment': sentiment, 'confidence': predictions.max().item()}

if __name__ == '__main__':
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze('This product is amazing!')
    print(result)
