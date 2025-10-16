#utils/visualizer.py
import spacy
from spacy import displacy

def visualize_entities(text):
    """Display named entities using SpaCy visualization with white text for dark theme."""
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    # Render entities as HTML
    html = displacy.render(doc, style='ent', page=True)
    
    # Inject custom CSS for white text
    custom_style = """
    <style>
        body {
            background-color: transparent;
            color: white;
            font-size: 1.1rem;
            line-height: 1.5;
        }
        mark {
            background: #3b82f6;  /* blue highlight */
            color: white !important;
            padding: 0.2em 0.4em;
            border-radius: 0.25em;
        }
        .entity {
            color: white !important;
        }
    </style>
    """
    
    return custom_style + html
