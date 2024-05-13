# dsml_project
Repository for the Data Science and Machine Learning course kaggle competition - Beatrice Vaienti and Chiara Berretta

# To Do - Preliminary evaluation
report the following table without doing any cleaning on the data. Do hyper-parameter optimization to find the best solution. Your code should justify your results.

table:
|      | Logistic Regression | KNN | Decision Tree | Random Forest | Other technique |
|------|--------------------|-----|--------------|---------------|-----------------|
| Precision | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Recall | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| F1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Accuracy | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Answer the following questions
- Which is the best model?
- Show the confusion matrix.
- Show examples of some erroneous predictions. Can you understand where the error is coming from?
- Do some more analysis to better understand how your model behaves.

# Ideas to try / references
https://medium.com/tech-tavern/nlp-exercise-text-difficulty-prediction-with-ai-ccefe99daa65



# Folder Structure
In our project, we explored multiple machine learning solutions to identify the most effective approach:
1. CamemBERT model
2. Flaubert model
3. A hybrid approach combining the best model of the previous two with a neural network on augmented data (the attributes derived from the text).

The organization of the code is as follows:
- The `models` folder contains the code for initializing and configuring models.
- The `utils` folder includes utilities for data preprocessing and model evaluation.
- The `scripts` folder houses scripts to evaluate, train, and make predictions with the models.

Saved hyperparameters and logs are stored in the `best_hyperparameters_saved` folder.
Trained models are saved in the `models_saved` folder.

# Flaubert / CamemBERT Model Training and Evaluation
We used the Hugging Face library to load pre-trained models and fine-tune them on our dataset. Our approach supports using either the CamemBERT or Flaubert model, selectable via command line.

## Scripts
- `evaluate_bert.py`: Manages training and validation loops, performing hyperparameter tuning with a train-validation split.
- `train_bert.py`: Trains a selected model on the full dataset.
- `predict_bert.py`: Makes predictions on new, unlabeled data using a trained model.

## Hyperparameter Tuning and Evaluation
Hyperparameter tuning is performed using `evaluate_bert.py` with a grid search over predefined values for learning rates, batch sizes, and epochs. Each configuration is evaluated on the validation set, and the best-performing parameters are recorded. Results are saved in the `best_hyperparameters_saved` folder.
Due to the computational cost of hyperparameter tuning, we opted to perform the evaluation with a simple train-validation split of 20%, without k-fold cross-validation.

To conduct hyperparameter tuning and evaluation, run:
```bash
python scripts/evaluate_bert.py --model [camembert|flaubert]
```

### Evaluation Results

The best hyperparameters discovered during the tuning process are detailed in the output files within the `best_hyperparameters_saved` folder. The log with all the combinations of hyperparameters and their corresponding validation accuracy is saved in the `best_hyperparameters_saved` folder.

#### Hyperparameter Tuning Log
The log file contains the validation accuracy for each combination of hyperparameters tested during the tuning process. The best hyperparameters are selected based on the highest validation accuracy achieved. In the following table, we show the accuracy obtained for each combination of hyperparameters tested and for each model.
##### CamemBERT
| Learning Rate | Batch Size | Epochs | Validation Accuracy |
|---------------|------------|--------|---------------------|
| Placeholder   | Placeholder| Placeholder | Placeholder      |

##### Flaubert
| Learning Rate | Batch Size | Epochs | Validation Accuracy |
|---------------|------------|--------|---------------------|
| Placeholder   | Placeholder| Placeholder | Placeholder      |


In the following table, we summarize the best validation accuracy achieved for each model with the best hyperparameters found.

#### Hyperparameter Tuning Table
| Model      | Learning Rate | Batch Size | Epochs | Validation Accuracy |
|------------|---------------|------------|--------|---------------------|
| CamemBERT  | Placeholder   | Placeholder| Placeholder | Placeholder      |
| Flaubert   | Placeholder   | Placeholder| Placeholder | Placeholder      |

#### Confusion Matrices
The confusion matrices for the CamemBERT and Flaubert models obtained for the best hyperparameters are shown below. 
![CamemBERT Confusion Matrix](path_to_camembert_confusion_matrix.png)
![Flaubert Confusion Matrix](path_to_flaubert_confusion_matrix.png)


## Training and Prediction
To train a model on the full dataset, execute:
```bash
python scripts/train_bert.py --model [camembert|flaubert]
```
The trained model will be saved in the `models_saved` folder.

To make predictions on the test set using a trained model, run:
```bash
python scripts/predict_bert.py --model [camembert|flaubert]
```
The model contained in the `models_saved` folder will be used to predict on the inference set, with results saved in the `predictions` folder.

# Combined Model Training and Evaluation
We combined the best-performing model from the CamemBERT and Flaubert approaches with a neural network trained on augmented data. The neural network was trained on the attributes derived from the text, in particular the number of words, the average length of the words, the POS tags.

# Accuracies obtained for the Kaggle competition

| Model      | Accuracy | 
|------------|----------|
| CamemBERT  | Placeholder | 
| Flaubert   | Placeholder |

# Conclusion






# Streamlit App
We created a Streamlit app to visualize the predictions of the models. The app allows users to input a text and see the predictions of the CamemBERT and Flaubert models. The app can be run locally by executing the following command:
```bash
streamlit run app.py
```
The app will open in a new browser window, and users can interact with it by entering text and seeing the model predictions. 
