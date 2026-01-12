# Steam-Game-Popularity-Prediction

## Problem Description
The goal of this project is to predict whether a Steam game will be successful based on early data and release information. Success is defined as having more than 0 user recommendations. This is a highly imbalanced problem: approximately 87.8% of games in the dataset have 0 recommendations, while only 12.2% have more than 0. 

**Note:** Defining success as having more than 0 user recommendations is a simplification driven by the dataset structure and is not an ideal representation of true game success.

Accurately predicting game success can help developers and publishers make data-driven decisions, such as prioritizing marketing, adjusting pricing strategies, or identifying games with high potential early in the release cycle.

---

## Dataset
The dataset used comes from Kaggle: [Steam Games Dataset 2021–2025 (65k games)](https://www.kaggle.com/datasets/jypenpen54534/steam-games-dataset-2021-2025-65k).  
It includes metadata such as:  
- Price  
- Developer and publisher information  
- Categories and genres  
- Release date 

### Instructions to access the dataset
1. Visit the Kaggle dataset page linked above.  
2. Download the dataset and rename it to `steam_data.csv`.  
3. Place the dataset in the directory of the project.

---

## Exploratory Data Analysis (EDA)
Extensive EDA was performed to understand the dataset and the target variable:

- Checked for missing values and ranges of numeric features  
- Analyzed the distribution of the target variable (successful vs unsuccessful games)  
- Conducted feature importance analysis using a trained neural network:

| Feature              | Importance |
|---------------------|------------|
| price                | 0.041      |
| publisher_encoded    | 0.041      |
| category             | 0.038      |
| num_categories       | 0.038      |
| genre                | 0.027      |
| developer_encoded    | 0.013      |
| release_month        | 0.0004     |
| release_year         | 0.00009    |
| release_day_of_week  | 0.00004    |
| num_genres           | -0.00013   |

---

## Model Training
Multiple models were trained and evaluated with hyperparameter tuning. AUC was used as the primary evaluation metric during model tuning because the dataset is highly imbalanced. Unlike accuracy, AUC measures a model’s ability to distinguish between successful and unsuccessful games across all classification thresholds, making it more robust for comparing models

| Model                | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------|----------|-----------|--------|-------|-------|
| Logistic Regression  | 0.814    | 0.380     | 0.762  | 0.507 | 0.869 |
| Decision Tree        | 0.724    | 0.279     | 0.758  | 0.408 | 0.798 |
| Random Forest        | 0.816    | 0.383     | 0.760  | 0.509 | 0.859 |
| Gradient Boosting    | 0.778    | 0.332     | 0.764  | 0.463 | 0.824 |
| Neural Network       | 0.860    | 0.465     | 0.759  | 0.577 | 0.901 |

The evaluation threshold was chosen to maintain recall > 0.75 while maximizing precision. The final selected model was a neural network.

---

## Replicating the Project
1. **Requirements:**
    * Ensure Docker is installed on your machine  

2. **Clone the repository:**
    ```
    git clone https://github.com/DylanD-H/Steam-Game-Popularity-Prediction.git
    cd Steam-Game-Popularity-Prediction
    ```
3. **Build the Docker image:**
    ```
    docker build -t steam-prediction .
    ```
    The Dockerfile copies all necessary files and dependencies into the container. The trained model is already included. If you prefer to train the model yourself, you can run:
   ```
   python scripts/train.py
   ```
4. **Run the Docker container:**
   ```
   docker run -it --rm -p 9696:9696 steam-prediction
   ```
5. **Test the model:**
   
   From your local machine, run:
   ```
   python scripts/game.py
   ```
   This will return the predicted probability that the game will be successful, along with the final decision based on the threshold.
