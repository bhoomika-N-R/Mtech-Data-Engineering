# Product Classification Using Machine Learning

## Project Overview

This project develops a machine learning model to automatically classify e-commerce products into predefined product categories using their product titles.

The project uses TF-IDF for text feature extraction and Logistic Regression for multiclass classification.

## Problem Statement

E-commerce platforms contain thousands of product listings from different merchants. Product titles may use different descriptions, model numbers and specifications, making manual categorization difficult.

The objective is to build a supervised machine learning model that predicts the correct product category from a product title.

## Dataset

The project uses the **Product Classification and Clustering** dataset from the UCI Machine Learning Repository.

- Total records: 35,311
- Features: 7
- Product categories: 10
- Merchants: 306
- Target variable: `Category Label`
- Input feature: `Product Title`

Dataset source: https://doi.org/10.24432/C5M91Z

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

Dataset citation:

Akritidis, L. (2020). Product Classification and Clustering [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5M91Z.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Workflow

1. **Data Understanding and Exploration**
   - Loaded and inspected the dataset.
   - Examined category distributions and unique values.
   - Investigated duplicate product titles and product clusters.

2. **Feature and Target Selection**
   - Input: Product Title
   - Target: Category Label
   - Cluster ID was used to keep related product offers together during splitting, not as a model input.

3. **Train-Test Split**
   - Used GroupShuffleSplit.
   - Training records: 28,157
   - Testing records: 7,154
   - Verified that no product clusters overlapped between the two sets.

4. **Text Feature Extraction**
   - Applied TF-IDF (Term Frequency–Inverse Document Frequency).
   - Used unigrams and bigrams.
   - Limited the vocabulary to 50,000 features.

5. **Model Training**
   - Algorithm: Logistic Regression
   - Solver: SAGA
   - Task: Multiclass classification

6. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
   - Misclassification analysis

## Model Results

| Metric | Result |
|---|---:|
| Accuracy | 94.94% |
| Macro Precision | 0.95 |
| Macro Recall | 0.95 |
| Macro F1-score | 0.95 |
| Misclassified test records | 362 |

The model achieved 94.94% accuracy on the held-out test set.

CPUs achieved an F1-score of 1.00, while Fridges had the lowest F1-score at 0.86.

## Error Analysis

The model made 362 incorrect predictions out of 7,154 test records.

Some observed challenges included:

- Product titles containing only brand names and model numbers.
- Similar technical identifiers across product categories.
- Ambiguous or unusual product descriptions.
- Confusion between related appliance categories, particularly Fridges and Fridge Freezers.

## Limitations and Future Improvements

- Experiment with additional text preprocessing techniques.
- Compare Logistic Regression with other text classification algorithms such as Linear SVM.
- Investigate ambiguous and incorrectly labelled product records.
- Explore improved handling of product model numbers and technical identifiers.

## Project Structure

```text
Product_Classification/
├── data/
│   └── pricerunner_aggregate.csv
├── product_classification.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `product_classification.ipynb` and run the cells in order.

## Conclusion

This project demonstrates an end-to-end supervised machine learning workflow for e-commerce product classification. TF-IDF and Logistic Regression achieved 94.94% test accuracy across 10 product categories.

The error analysis also highlighted challenges associated with ambiguous product titles and overlapping category descriptions.