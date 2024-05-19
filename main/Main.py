import gradio as gr
import whisper
import spacy
from pydub import AudioSegment
import io

# Load Whisper model
model = whisper.load_model("tiny")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def transcribe(audio):

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # Get the transcription text
    transcription = result.text

    # Analyze the transcription text using spaCy
    entities_html, pos_tags_text, dependency_info_text, sentences_text = analyze_text(
        transcription)

    # Return the transcription and the NLP insights
    return transcription, entities_html, pos_tags_text, dependency_info_text, sentences_text


def analyze_text(text):
    if not text:
        return "", "", "", ""
    try:
        doc = nlp(text)
    except Exception as e:
        # Handle or log the exception as needed
        return "", "", "", ""

    try:
        # Named Entities Visualization
        entities_html = spacy.displacy.render(doc, style="ent", page=True)
    except Exception as e:
        # Handle or log the exception as needed
        entities_html = ""

    try:
        # Part-of-speech (POS) Tagging
        pos_tags = [(token.text, token.pos_) for token in doc]
        pos_tags_text = "\n".join([f"{text}: {pos}" for text, pos in pos_tags])
    except Exception as e:
        # Handle or log the exception as needed
        pos_tags_text = ""

    try:
        # Dependency Parsing
        dependency_info = [(token.text, token.dep_, token.head.text)
                           for token in doc]
        dependency_info_text = "\n".join(
            [f"{text} --{dep}--> {head}" for text, dep, head in dependency_info])
    except Exception as e:
        # Handle or log the exception as needed
        dependency_info_text = ""

    try:
        # Sentence Segmentation
        sentences = [sent.text for sent in doc.sents]
        sentences_text = "\n".join(sentences)
    except Exception as e:
        # Handle or log the exception as needed
        sentences_text = ""

    return entities_html, pos_tags_text, dependency_info_text, sentences_text


interface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Upload Audio File")
    ],
    outputs=[
        "textbox",
        gr.HTML(label="Named Entities Visualization"),
        gr.Textbox(label="POS Tagging"),
        gr.Textbox(label="Dependency Parsing"),
        gr.Textbox(label="Sentence Segmentation")
    ],
    title="Whisper-based ASR Model with NLP Insights",
    description="This application takes a 30-second audio input from an uploaded file, transcribes it using OpenAI Whisper, and generates insights using spaCy."
)

interface.launch(share=True)
