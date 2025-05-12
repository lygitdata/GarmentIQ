# Model fine-tuning briefing

Date: 05/13/2025

Ranking:

- 🏆 1st place: tinyViT (`tiny_vit_inditex_finetuned.pt`) - execellent performance on new Zara data (99.16% accuracy), preserves strong performance on the original Nordstrom & Myntra testing data (94.84% accuracy).

- 🥈 2nd place: CNN4 (`cnn_4_inditex_finetuned.pt`)

- 🥉 3rd place: CNN3 (`cnn_3_inditex_finetuned.pt`)

Data:

- Zara data: https://www.kaggle.com/datasets/lygitdata/zara-clothes-image-data

- Nordstrom & Myntra data: https://www.kaggle.com/datasets/lygitdata/garmentiq-classification-set-nordstrom-and-myntra

---

## tinyViT performance with fine-tuning

### Test on Zara data

```
Test Loss: 0.0390
Test Accuracy: 0.9916
Test F1 Score: 0.9917

Classification Report:
                    precision    recall  f1-score   support

 long sleeve dress       1.00      0.99      0.99        80
   long sleeve top       0.99      0.99      0.99       352
short sleeve dress       0.93      1.00      0.97        14
  short sleeve top       1.00      1.00      1.00        84
            shorts       1.00      1.00      1.00        56
             skirt       1.00      1.00      1.00        62
          trousers       1.00      1.00      1.00        80
              vest       0.96      0.98      0.97        94
        vest dress       1.00      1.00      1.00        12

          accuracy                           0.99       834
         macro avg       0.99      0.99      0.99       834
      weighted avg       0.99      0.99      0.99       834
```

### Test on original testing data

```
Test Loss: 0.1683
Test Accuracy: 0.9484
Test F1 Score: 0.9483

Classification Report:
                    precision    recall  f1-score   support

 long sleeve dress       0.89      0.95      0.92       384
   long sleeve top       0.94      1.00      0.97       442
short sleeve dress       0.90      0.88      0.89       382
  short sleeve top       0.98      0.98      0.98       523
            shorts       0.99      0.97      0.98       485
             skirt       0.98      0.90      0.94       281
          trousers       0.97      0.99      0.98       320
              vest       0.95      0.89      0.92       230
        vest dress       0.94      0.93      0.93       442

          accuracy                           0.95      3489
         macro avg       0.95      0.94      0.95      3489
      weighted avg       0.95      0.95      0.95      3489
```

---

## CNN4 performance with fine-tuning

### Test on Zara data

```
Test Loss: 0.1355
Test Accuracy: 0.9592
Test F1 Score: 0.9585

Classification Report:
                    precision    recall  f1-score   support

 long sleeve dress       0.98      0.99      0.98        80
   long sleeve top       0.99      0.97      0.98       352
short sleeve dress       0.75      0.86      0.80        14
  short sleeve top       0.92      0.98      0.95        84
            shorts       0.96      0.88      0.92        56
             skirt       0.88      0.97      0.92        62
          trousers       0.98      1.00      0.99        80
              vest       0.97      0.97      0.97        94
        vest dress       0.86      0.50      0.63        12

          accuracy                           0.96       834
         macro avg       0.92      0.90      0.90       834
      weighted avg       0.96      0.96      0.96       834
```

### Test on original testing data

```
Test Loss: 0.3326
Test Accuracy: 0.9132
Test F1 Score: 0.9137

Classification Report:
                    precision    recall  f1-score   support

 long sleeve dress       0.92      0.83      0.87       384
   long sleeve top       0.97      0.88      0.92       442
short sleeve dress       0.75      0.97      0.84       382
  short sleeve top       0.91      0.97      0.94       523
            shorts       0.94      0.98      0.96       485
             skirt       0.95      0.83      0.88       281
          trousers       0.96      0.97      0.97       320
              vest       0.95      0.85      0.90       230
        vest dress       0.94      0.88      0.91       442

          accuracy                           0.91      3489
         macro avg       0.92      0.91      0.91      3489
      weighted avg       0.92      0.91      0.91      3489
```

---

## CNN3 performance with fine-tuning

### Test on Zara data

```
Test Loss: 0.2756
Test Accuracy: 0.9197
Test F1 Score: 0.9216

Classification Report:
                    precision    recall  f1-score   support

 long sleeve dress       0.89      0.95      0.92        80
   long sleeve top       0.97      0.95      0.96       352
short sleeve dress       0.62      0.71      0.67        14
  short sleeve top       0.94      0.95      0.95        84
            shorts       0.94      0.86      0.90        56
             skirt       0.79      0.92      0.85        62
          trousers       0.99      1.00      0.99        80
              vest       0.92      0.82      0.87        94
        vest dress       0.35      0.50      0.41        12

          accuracy                           0.92       834
         macro avg       0.82      0.85      0.83       834
      weighted avg       0.93      0.92      0.92       834
```

### Test on original testing data

```
Test Loss: 0.3338
Test Accuracy: 0.9074
Test F1 Score: 0.9068

Classification Report:
                    precision    recall  f1-score   support

 long sleeve dress       0.84      0.89      0.86       384
   long sleeve top       0.94      0.93      0.94       442
short sleeve dress       0.76      0.94      0.84       382
  short sleeve top       0.93      0.98      0.95       523
            shorts       0.96      0.97      0.96       485
             skirt       0.94      0.81      0.87       281
          trousers       0.94      0.99      0.97       320
              vest       0.99      0.69      0.81       230
        vest dress       0.94      0.84      0.89       442

          accuracy                           0.91      3489
         macro avg       0.92      0.89      0.90      3489
      weighted avg       0.91      0.91      0.91      3489
```
