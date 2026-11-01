# Forest Cover Type Classification with Deep Learning

## Project Overview

This project uses **TensorFlow and Keras** to predict forest cover type from cartographic and environmental variables collected from the Roosevelt National Forest in northern Colorado.

The objective was to build, evaluate, and optimize a multi-class neural network capable of classifying one of seven forest cover types using topographical, hydrological, wilderness area, and soil information.

The project evolved beyond a simple deep learning exercise into a full machine learning workflow involving:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature scaling
- Neural network development
- Hyperparameter tuning
- Model comparison
- Error analysis
- Model persistence
- Performance reporting

---

## Business Problem

Accurately identifying forest cover types is important for:

- Forest management
- Conservation planning
- Environmental monitoring
- Ecological research
- Resource allocation

Rather than relying solely on manual classification, this project demonstrates how deep learning can leverage geographical and environmental measurements to automate cover type prediction.

---

## Dataset

### Source

US Forest Service (USFS) Region 2 Resource Information System data combined with cartographic variables derived from:

- US Geological Survey (USGS)
- United States Forest Service (USFS)

### Dataset Characteristics

| Metric | Value |
|----------|----------:|
| Rows | 581,012 |
| Features | 54 |
| Target Classes | 7 |
| Missing Values | 0 |
| Duplicate Rows | 0 |

### Target Classes

| Class | Forest Cover Type |
|---------|---------|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## Data Preparation

The preprocessing pipeline included:

- Data quality assessment
- Missing value verification
- Duplicate record detection
- Feature categorization
- Train/validation/test splitting
- Numerical feature standardization
- Target encoding for TensorFlow

### Feature Groups

#### Numerical Features

- Elevation
- Aspect
- Slope
- Hydrology distances
- Roadway distances
- Fire point distances
- Hillshade measurements

#### Binary Features

- Wilderness area indicators
- Soil type indicators

### Dataset Split

| Dataset | Samples |
|----------|----------:|
| Training | 406,708 |
| Validation | 87,152 |
| Test | 87,152 |

Stratified sampling was used to maintain class distributions across all datasets.

---

## Exploratory Data Analysis

Key findings:

### Data Quality

✅ No missing values

✅ No duplicate observations

✅ Clean and well-structured dataset

### Class Imbalance

The dataset contained significant class imbalance.

- Class 1 and Class 2 represented approximately 85% of observations
- Class 4 represented less than 1% of observations

This made evaluation metrics beyond accuracy particularly important.

---

## Baseline Model

### Architecture

```python
Dense(64, activation="relu")
Dense(32, activation="relu")
Dense(7, activation="softmax")
```

### Results

| Metric | Score |
|----------|----------:|
| Validation Accuracy | 85.64% |
| Test Accuracy | 85.55% |

The baseline model established a strong benchmark with minimal overfitting.

---

## Hyperparameter Tuning

A series of controlled experiments were conducted to evaluate the effect of different hyperparameters.

| Model | Experiment | Validation Accuracy |
|---------|---------|----------:|
| Model 1 | Baseline | 85.64% |
| Model 2 | Dropout + Early Stopping | 80.39% |
| Model 3 | Larger Network | 89.84% |
| Model 4 | Learning Rate = 0.0005 | 88.77% |
| Model 5 | RMSprop Optimizer | 85.26% |
| Model 6 | Batch Size = 64 | **90.05%** |
| Model 7 | ELU Activation | 89.56% |

### Most Important Finding

The largest improvement came from increasing network capacity:

```python
Dense(128, activation="relu")
Dense(64, activation="relu")
Dense(32, activation="relu")
Dense(7, activation="softmax")
```

Additional regularization and optimizer changes did not outperform the larger architecture.

---

## Final Model

### Architecture

```python
Dense(128, activation="relu")
Dense(64, activation="relu")
Dense(32, activation="relu")
Dense(7, activation="softmax")
```

### Training Configuration

| Hyperparameter | Value |
|----------|----------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Activation Function | ReLU |
| Loss Function | Sparse Categorical Crossentropy |

---

## Final Results

| Metric | Score |
|----------|----------:|
| Validation Accuracy | 90.05% |
| Test Accuracy | 90.00% |
| Weighted F1 Score | 0.90 |
| Macro F1 Score | 0.85 |

### Improvement Over Baseline

```text
Baseline Test Accuracy: 85.55%
Final Test Accuracy:    90.00%

Improvement: +4.45%
```

---

## Classification Performance

### Strongest Classes

| Class | F1 Score |
|----------|----------:|
| Class 6 | 0.92 |
| Class 1 | 0.91 |
| Class 0 | 0.90 |

### Most Challenging Classes

| Class | F1 Score |
|----------|----------:|
| Class 3 | 0.77 |
| Class 4 | 0.76 |

The weakest classes were also among the least represented in the dataset.

---

## Error Analysis

The confusion matrix revealed that the most common classification errors occurred between:

```text
Class 0 ↔ Class 1
```

This suggests these forest cover types share similar environmental and cartographic characteristics, making them difficult to distinguish using the available variables.

Despite significant class imbalance, the model maintained strong precision and recall across all classes.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

```text
forest-cover-type-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── final_forest_cover_model.keras
│
├── notebooks/
│   └── forest_cover_analysis.ipynb
│
├── reports/
│
├── src/
│
├── images/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Key Takeaways

- Deep learning can effectively classify forest cover types from cartographic variables.
- Increasing model capacity provided the largest performance gains.
- Additional regularization was unnecessary because the baseline model exhibited minimal overfitting.
- Systematic hyperparameter tuning improved test accuracy by approximately 4.5 percentage points.
- The final model demonstrated strong generalization with a test accuracy of 90.00%.

---

## Future Improvements

Potential enhancements include:

- Class weighting
- SMOTE oversampling
- Bayesian hyperparameter optimization
- Ensemble methods
- Advanced tabular deep learning architectures
- Additional feature engineering

---

## Author

Developed as part of a deep learning portfolio project using TensorFlow and Keras, demonstrating the complete machine learning workflow from data preparation through model deployment readiness.