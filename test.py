from flask import Flask, render_template, request
from pydoc import doc
import assemblyai as aai
import re
import spacy

# Replace with your AssemblyAI API key
aai.settings.api_key = "f47475f73886450aa9936c355dfbb335"  # Insert your AssemblyAI API key here

app = Flask(__name__,template_folder="templates")

def transcribe_local_file(filepath):
    try:
        # Create a Transcriber object
        transcriber = aai.Transcriber()

        # Upload the local audio file using the upload_file method
        upload_url = transcriber.upload_file(filepath)

        # Configure transcription (optional)
        config = aai.TranscriptionConfig(speaker_labels=True)  # Uncomment for speaker labels

        # Transcribe the uploaded audio
        transcript = transcriber.transcribe(upload_url, config)

        # Extract and return the transcribed text
        return transcript.text

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None  # Indicate error

def identify_symptoms(transcribed_text, symptom_vocab):
    symptoms = set()  # Use a set to store unique symptoms
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(transcribed_text)

    for token in doc:
        if token.pos_ in ("NOUN", "VERB"):  # Focus on nouns and verbs for symptom terms
            if token.text.lower() in symptom_vocab:
                symptoms.add(token.text)  # Add unique symptom to the set

    return list(symptoms)  # Convert the set back to a list

symptom_vocab = {
    "headache": {"pos": "NOUN", "tag": "SYMPTOM"},
    "pain": {"pos": "NOUN", "tag": "SYMPTOM"},
    "nausea": {"pos": "NOUN", "tag": "SYMPTOM"},
    "vomiting": {"pos": "VERB", "tag": "SYMPTOM"},
    "stiffness": {"pos": "NOUN", "tag": "SYMPTOM"},
    "dizziness": {"pos": "NOUN", "tag": "SYMPTOM"},
    "fever": {"pos": "NOUN", "tag": "SYMPTOM"},  # Optional additions
    "cough": {"pos": "NOUN", "tag": "SYMPTOM"},  # Optional additions
}

def extract_info(transcribed_text):
    names = []  # List to store names (optional)
    age = None

    # Use spaCy for potential name recognition (optional)
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(transcribed_text)
    for entity in doc.ents:
        if entity.label_ == "PERSON":  # Consider using a more specific label for names if needed
            names.append(entity.text)

    # Extract age using pattern matching
    age_match = re.search(r"\d+ (years old|yo)", transcribed_text)
    if age_match:
        age = int(age_match.group().split()[0])

    return {"names": names, "age": age}

@app.route("/", methods=["GET", "POST"])

def transcribe_and_analyze():
    if request.method == "POST":
        uploaded_file = request.files["audio_file"]
        transcribed_text = transcribe_local_file(uploaded_file.filename)
        if transcribed_text:
            extracted_info = extract_info(transcribed_text)
            symptoms = identify_symptoms(transcribed_text, symptom_vocab)
            return render_template("results.html",transcribed_text=transcribed_text, extracted_info=extracted_info, symptoms=symptoms)
        else:
                return "Error during transcription."
            
    else:
         return render_template("index.html")
    
    



    



if __name__ == "__main__":
    app.run(debug=True)
