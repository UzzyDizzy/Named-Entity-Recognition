#utils/visualizer.py
import spacy
from spacy import displacy
import subprocess
import importlib

# Ensure the model is available in the environment
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    importlib.invalidate_caches()
    nlp = spacy.load("en_core_web_sm")



def visualize_entities(text):
    """Visualize named entities with white text and color-coded highlights."""
    doc = nlp(text)
    colors = {
        "PERSON": "#4DD0E1",
        "ORG": "#81C784",
        "GPE": "#FFB74D",
        "LOC": "#E57373",
        "DATE": "#BA68C8"
    }
    html = displacy.render(doc, style="ent", options={"colors": colors})
    styled_html = f"""
    <style>
        body {{ color: white; background-color: transparent; font-size: 1.1rem; }}
        mark {{ color: white !important; padding: 0.2em 0.4em; border-radius: 0.25em; }}
    </style>
    {html}
    """
    return styled_html
