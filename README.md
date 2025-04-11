# IFSSAA Client Retention Model

## Overview
This project aims to build a client retention model for the IFSSA food hamper distribution program. The model predicts whether a client will be retained ("Yes") or churned ("No") based on historical interaction data and environmental influences. The goal is to identify at-risk clients, allowing the program to tailor outreach strategies and services effectively.

## Project Objectives
- Perform EDA to get insights on clients behaviour and patterns
- Create more features and target column
- Develop a client retention classification model
- Develop RAG chatbot
- Deploy prediction application 

## Project Deliverables
The project involves two major deliverables                                                                            
1. Client-Retention Classifiation
2. Retrieval-Augmented Generation (RAG)

## Project Phases

1. Business Ideation and Stakeholder Meeting    
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Development and Optimization
5. Model Prediction Explainability (XAI)
6. RAG implementation                                                         7. Model Deployment
   
## Features
The model utilizes the following engineered features:
1. **return_binary**: Target column indicating client return status within a 60-day window.
2. **days_since_last_pickup**: Tracks client interaction frequency.
3. **days_diff_scheduled_actual**: Measures service punctuality.
4. **rescheduled_flag**: Indicates if a pickup was rescheduled.
5. **month**: Captures seasonal trends.
6. **total_visits**: Sums visits per neighborhood.
7. **avg_days_between_pickups**: Calculates average gap between visits.
8. **is_single_pickup**: Flags clients with only one pickup.
9. **distance_to_center**: Measures distance from pickup location to client address.
10. **location_cluster**: Enables spatial analysis for resource optimization.



## Model Development
### Model Selection
Seven classification models were evaluated:
1. CatBoost
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. LightGBM
5. Random Forest
6. Gradient Boosting
7. XGBoost

### Best Performing Model
**CatBoost** achieved the highest F1-score of **0.9269** and was selected for further optimization.

### Hyperparameter Tuning
RandomizedSearchCV was used to fine-tune CatBoost, resulting in the following optimal parameters:
- 'classifier__learning_rate': 0.05500000000000001,
- 'classifier__l2_leaf_reg': 5,
- 'classifier__iterations': 1000,
- 'classifier__depth': 8

The tuned model achieved an F1-score of **0.9292** on the test set.

## Model Deployment
The model is deployed via a Streamlit web app, allowing users to input client details and receive predictions. The app includes:
- **Dashboard**: Overview and navigation.
- **Insights**: Exploratory data analysis and visualizations.
- **Predictions**: Interface for inputting client data and viewing predictions.
- **Chatbot**: Interface for to get more insights about the project.

## Explainable AI (XAI)
SHAP (SHapley Additive exPlanations) was used to interpret the model's predictions, providing insights into feature importance and individual prediction explanations.

## Usage
### Prerequisites
- Python 3.11
- Libraries listed in `requirements.txt`

### Running the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Files
- `best_catboost_model.pkl`: Saved tuned CatBoost model.
- `model_top7.pkl`: Model trained on top 7 features.
- `feature_importances.csv`: Feature importance rankings.
- `app.py`: Streamlit application script.

## Results
- **Top Features**: `month `total_visits`, `month`, `avg_days_between_pickups`, `days_since_last_pickup`, 'distance_to_center', 'dependents_qty', 'location_cluster_.
- **Test Performance**:
  - Precision: 0.90
  - Recall: 0.97
  - F1-score: 0.93
  - ROC AUC: 0.91
 
## Deployed Application Link:
[Client Retention Classification App](https://ifssa-client-retention.streamlit.app/)   

## Project Visualizations
[Insights and Visualizations](https://www.notion.so/Client-Retention-Classification-1cb67eaaf9588085b49dc31c751614a6)                                  

## For a full description of the project and timelines please refer to the following link:
[Notion Project Overview and Timelines](https://www.notion.so/Client-Retention-Classification-1cb67eaaf9588085b49dc31c751614a6)

## To run the project codes

[Exploratory Data Analysis Notebook](https://colab.research.google.com/drive/1Ao01Q0fzSF8XMfinILx9A7-P7364Gz1s)

[Feature Engineering Notebook](https://colab.research.google.com/drive/13VHC4WsE6DK7uzgOGnM9gU7IWQY-2d3l?usp=sharing)

[Model Development, Optimization and Deployment](https://colab.research.google.com/drive/179UCwxtkTwZw1zHjrarTgxsGzXxDXql9?usp=sharing)

## Contributors
- Chioma [Linkedin](https://www.linkedin.com/in/chioma-ulom-82344360/utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app)  [GitHub](https://github.com/ChiomaUU)
- Enkeshie  [Linkedin](https://www.linkedin.com/in/enkeshie-parris-3406b7259/ )     [Github](https://github.com/KayLiz)
- Renata Saccon [Linkedin](https://www.linkedin.com/in/renata-saccon/)        [Github](https://github.com/RSacconNQ)
- Ruth [Linkedin](https://www.linkedin.com/in/rutholasupo/)       [Github](https://github.com/RuthOlasupo)



## License
MIT

---

For detailed code and implementation, refer to the Jupyter notebook and scripts in the repository.
