import streamlit as st
from utils.visualizer import visualize_entities

st.set_page_config(page_title="NER-Identifier", layout="wide")

st.title("🧠 Named Entity Recognition (NER) using BiLSTM-LSTM")
st.markdown("Identify people, organizations, locations, and more using a trained deep learning model.")

# ---- SIDEBAR: Show full forms for abbreviations ----
st.sidebar.header("📘 NER Tag Full Forms")

entity_fullforms = {
    "PERSON": "Person (individual, including fictional)",
    "ORG": "Organization (companies, institutions, agencies)",
    "GPE": "Geo-Political Entity (countries, cities, states)",
    "LOC": "Location (non-GPE, mountains, bodies of water)",
    "PRODUCT": "Product (objects, vehicles, foods, etc.)",
    "EVENT": "Event (named hurricanes, wars, sports events)",
    "WORK_OF_ART": "Work of Art (titles of books, songs, etc.)",
    "LANGUAGE": "Language (named natural or constructed languages)",
    "DATE": "Date or period",
    "TIME": "Time smaller than a day",
    "PERCENT": "Percentage (including “%”)",
    "MONEY": "Monetary values, including units",
    "QUANTITY": "Measurements, as of weight or distance",
    "ORDINAL": "Ordinal number (first, second, etc.)",
    "CARDINAL": "Numerals that do not fall under another type",
}

for abbr, full in entity_fullforms.items():
    st.sidebar.markdown(f"**{abbr}** → {full}")

# ---- MAIN APP: Text analysis ----
st.subheader("🧾 Test the Model on Custom Text")

user_input = st.text_area(
    "Enter text here:",
    "Hi, my name is Aman and I want to work at Google in India."
)

if st.button("Analyze Entities"):
    st.markdown("### 🟢 Named Entities Visualization")
    html = visualize_entities(user_input)
    st.components.v1.html(html, height=400, scrolling=True)
