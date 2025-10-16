# 🧠 Named Entity Recognition (NER) using BiLSTM-LSTM

This project demonstrates Named Entity Recognition (NER) using a
BiLSTM-LSTM deep learning architecture, integrated with a Streamlit
interface for model training and testing.

------------------------------------------------------------------------

## ⚙️ Setup Instructions

### 1. Clone or Open the Project

``` bash
cd NAMED-ENTITY-RECOGNITION
```

### 2. Create and Activate Virtual Environment

``` bash
python -m venv myenv

# Activate on Windows
myenv\Scripts\activate

# Activate on Mac/Linux
source myenv/bin/activate
```

### 3. Install Requirements

``` bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

    streamlit
    pandas
    numpy
    tensorflow
    scikit-learn
    keras
    spacy

### 4. Download SpaCy Model

``` bash
python -m spacy download en_core_web_sm
```

### 5. Place Dataset

Place your dataset file here:

    data/ner_dataset.csv

------------------------------------------------------------------------

## 🧠 Train the Model

Run the training script:

``` bash
python train_model.py
```

This will: - Load and preprocess the dataset - Train a BiLSTM-LSTM
model - Save the trained model to `models/ner_model.h5`

------------------------------------------------------------------------

## 🚀 Run the Streamlit App

Launch the web interface:

``` bash
streamlit run app.py
```

Access it at: <http://localhost:8501>

------------------------------------------------------------------------

## 📁 Project Structure

    NAMED-ENTITY-RECOGNITION/
    │
    ├── data/
    │   └── ner_dataset.csv
    │
    ├── models/
    │   └── ner_model.h5
    │
    ├── utils/
    │   ├── data_loader.py
    │   ├── model_builder.py
    │   └── visualizer.py
    │
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🧩 Module Overview

  File                       Description
  -------------------------- ----------------------------------------------
  `utils/data_loader.py`     Loads and preprocesses dataset
  `utils/model_builder.py`   Defines BiLSTM-LSTM model architecture
  `utils/visualizer.py`      Displays entity visualization using SpaCy
  `train_model.py`           Trains and saves the model
  `app.py`                   Streamlit interface for training and testing

------------------------------------------------------------------------

## 🧾 Example

**Input:** \> "Hi, my name is Aman and I want to work at Google in
India."

**Output:** \| Entity \| Type \| \|---------\|------\| \| Aman \| PERSON
\| \| Google \| ORG \| \| India \| GPE \|

------------------------------------------------------------------------

## 🧑‍💻 Author

**Developed by:** UzzyDizzy\
A complete Named Entity Recognition (NER) web app built using
TensorFlow, Keras, and Streamlit.
