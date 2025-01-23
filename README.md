# 🤖 GLMMC: Generalist and Lightweight Model for Multilabel Classification 

GLMMC is a Multilabel Classification Model capable of classifying texts into various predefined entities using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to Large Language Models (LLMs), which, despite their flexibility, are costly and too large for resource-constrained scenarios.

### Usage
```python

from model import BiEncoderModel

texts = ["A celebrity chef has opened a new restaurant specializing in vegan cuisine.",
         "Doctors are warning about the rise in flu cases this season.",
         "The United States has announced plans to build a wall on its border with Mexico."]
batch_labels = [

        ["Food", "Business", "Politics"],
        ["Health", "Food", "Public Health"],
        ["Immigration", "Religion", "National Security"]
    ]
# Load the model
model = BiEncoderModel("sabdou/bi-encoder-model", max_num_labels=6)
# Prediction with JSON output
predictions = model.forward_predict(texts, batch_labels)
print("Predictions:", predictions)

```


#### Expected Output

```
Predictions: [
{'text': 'A celebrity chef has opened a new restaurant specializing in vegan cuisine.', 'scores': {'Food': 1.0, 'Business': 1.0, 'Politics': 0.0}},
{'text': 'Doctors are warning about the rise in flu cases this season.', 'scores': {'Health': 1.0, 'Food': 0.0, 'Public Health': 1.0}},
{'text': 'The United States has announced plans to build a wall on its border with Mexico.', 'scores': {'Immigration': 1.0, 'Religion': 0.0, 'National Security': 1.0}
]

```

#### Data

Synthetic data generated with gpt4-mini and gemini 

## Author 🧑‍💻
- [Salim ABDOU DAOURA](https://github.com/sabdoudaoura)
