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


# Folder Structure
In our project, we explored multiple machine learning solutions to identify the most effective approach:
1. CamemBERT model
2. Flaubert model
3. A hybrid approach combining the previous two with a neural network on the sentence embeddings and augmented data (the attributes derived from the text).

The organization of the code is as follows:
- The `models` folder contains the code for initializing and configuring models.
- The `utils` folder includes utilities for data preprocessing.
- The `scripts` folder houses scripts to evaluate, train, and make predictions with the models.

Saved hyperparameters and logs are stored in the `best_hyperparameters_saved` folder.
Trained models are saved in the `models_saved` folder.

# Flaubert / CamemBERT Model Training and Evaluation
We used the Hugging Face library to load pre-trained models and fine-tune them on our dataset. Our approach supports using either the CamemBERT or Flaubert model, selectable via command line. 

## Scripts
- `evaluate_bert.py`: Manages training and validation loops, performing hyperparameter tuning with a train-validation split.
- `train_bert.py`: Trains a selected model on the full dataset.
- `predict_bert.py`: Makes predictions on new, unlabeled data using a trained model.

In order to increase the batch size even without the computational resources, we used the __gradient accumulation technique__. This technique allows us to simulate a larger batch size by accumulating gradients over multiple steps before updating the model weights. In particular we set the gradient accumulation to calculate the steps in order to simulate a batch size of 64. However, we still evaluated the impact of different actual batch sizes, since, notwithstanding the gradient accumulation technique, the accuracy is still influenced by the actual batch size.

## evaluate_bert.py: Hyperparameter Tuning and Evaluation with Grid Search
Hyperparameter tuning is performed using `evaluate_bert.py` with a grid search over predefined values for learning rates, batch sizes, and epochs. Each configuration is evaluated on the validation set, and the best-performing parameters are recorded. Results are saved in the `hyperparameters_log` folder.
Due to the computational cost of hyperparameter tuning, we opted to perform the evaluation with a simple train-validation split of 20%, without k-fold cross-validation.

To conduct hyperparameter tuning and evaluation, run:
```bash
python scripts/evaluate_bert.py --model [camembert|camembert-large|flaubert]
```

The `--model` argument specifies the model to use, with options for `camembert-base` (camembert), `camembert-large`, and `flaubert-base`. The script will perform hyperparameter tuning and save a log with the results of the grid search in the `best_hyperparameters_saved` folder. Given the computational cost of hyperparameter tuning, we decided to carefully pick the hyperparameters to test each time after some preliminary tests, without repeating the same ones for all the models. 

### Evaluation Results
The logs with all the tested combinations of hyperparameters and their validation accuracy can be found within the `best_hyperparameters_saved` folder. 


#### Hyperparameter Tuning Log
The log file contains the validation accuracy for each combination of hyperparameters tested during the tuning process. The best hyperparameters are selected based on the highest validation accuracy achieved. In the following table, we show the accuracy obtained for each combination of hyperparameters tested and for each model.
##### CamemBERT
The model used for CamemBERT is the `camembert-base` model. 
As a first test, we evaluated the following hyperparameters:
- Learning Rate: [1e-05, 5e-05];
- Batch Size: [16, 32, 40, 45] with gradient accumulation to simulate a batch size of 64;
- Epochs: [10, 16, 20]


The results of this first experiment are shown in the following plot:

![CamemBERT Hyperparameter Tuning Log 1](images/grid_search_1_camembert.png)

From this first experiment, the learning rate 5e-05 appears to consistently outperform the lower learning rate of 1e-05. As such, we decided to focus on this learning rate for the subsequent experiments. Noticing a trend of increasing accuracy with more epochs and for higher batch sizes we decided to explore further the batch sizes of 40 and 45, adding a longer epoch of 32. The results of this second experiment are shown in the following plot:


![CamemBERT Hyperparameter Tuning Log 2](images/grid_search_2_camembert.png)






##### Flaubert
The model used for Flaubert is the `flaubert/flaubert_base_cased` model.

![Flaubert Hyperparameter Tuning Log 2](images/grid_search_flaubert.png)

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
