# Football Transfer Value Prediction

## Overview
This project predicts the expected **post-season transfer value** of football players based on pre-season attributes using the trained machine learning model. The prediction model uses pickle for serialization and requires specific input parameters to generate accurate market value forecasts.

The `train` is a single file and it contains multiple files so the `train` is a folder.

## Features
- Loads a pre-trained machine learning model from a pickle file
- Takes multiple player attributes as input
- Predicts future market value in millions of Euros


## Model File
The system requires a trained model file:
- File name: `trained_valuation_model.pkl`
- Location: Google drive

## Input Parameters
When running the model, users must provide the following 4 inputs:

1. **Previous Market Value (in millions)**: The player's market value before the current season in millions of Euros
2. **Score (1-100)**: An internal metric evaluating the player's performance or potential
3. **Age**: The player's age in years
4. **Cluster ID (1-8)**: A cluster label that categorizes the player based on similar characteristics or playing styles

## Output
The model returns:
- **Predicted Future Market Value (in millions of Euros)**: The forecasted market value of the player at the end of the season

## How to Run
1. Ensure the trained model file `trained_valuation_model.pkl` is in the same directory
2. Run the script: `python player_valuation.py`
3. Follow the prompts to enter player data
4. View the predicted market value

## Error Handling
The script includes error handling for:
- Missing model file
- Invalid input formats
- General exceptions during execution

## Sample Input/Output

**Input:**
- Previous Market Value: `1.5` million Euros
- Score: `78`
- Age: `24`
- Cluster ID: `2`

**Output:**
- **Predicted Future Market Value:** `1.8` million Euros

## Dataset and Models

- **Dataset**: [Google Drive - Football Player Dataset](https://drive.google.com/drive/folders/1CwpHjd-rPyuhg1bjW9U1kPMtsVraKdaT)
- **Models**: [Google Drive - Trained Models](https://drive.google.com/drive/folders/1RJ3zO1sd7LWviHWCI-3qtWCF8dXsrPZU)