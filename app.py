import torch
import gradio as gr
import json
from transformers import pipeline

# Load translation model
text_translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)

# Load language data
with open('language.json', 'r') as file:
    language_data = json.load(file)

def get_FLORES_code_from_language(language):
    for entry in language_data:
        if entry['Language'].lower() == language.lower():
            return entry['FLORES-200 code']
    return None

def translate_text(text, destination_language):
    dest_code = get_FLORES_code_from_language(destination_language)
    if not dest_code:
        return "Language code not found. Please try again."
    translation = text_translator(text, src_lang="eng_Latn", tgt_lang=dest_code)
    return translation[0]["translation_text"]

# Gradio layout with default theme
description_text = """
# Multi-Language Translator  
Translate English text into multiple languages easily and efficiently.  

### Credits:  
Made by Taizun S
"""

with gr.Blocks() as demo:
    gr.Markdown(description_text)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input English Text", placeholder="Type your text here...", lines=6)
            destination_language = gr.Dropdown(["German", "French", "Hindi", "Romanian", "Arabic"], 
                                               label="Select Language")
            submit_button = gr.Button("Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Translated Text", placeholder="Your translation appears here...", lines=4)

    # Link the components
    submit_button.click(fn=translate_text, inputs=[input_text, destination_language], outputs=[output_text])

demo.launch()
