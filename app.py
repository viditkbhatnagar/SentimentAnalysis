import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import json
import nltk
import whisper
import tempfile
import plotly.express as px
from transformers import pipeline, AutoTokenizer

# Download required NLTK resources
nltk.download('punkt')

# Initialize the Hugging Face sentiment analysis pipeline and tokenizer.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, device=-1)

# ===== PARAMETERS FOR SCALING (Model Calibration) =====
RATING_SCALING = 0.9  # Scale down counselor rating by 10%

# ================================
# HELPER FUNCTION: SPLIT TEXT INTO CHUNKS
# ================================
def split_text_into_chunks(text, max_tokens=512):
    """Splits the text into chunks with at most max_tokens tokens."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i: i + max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# ================================
# ANALYSIS FUNCTIONS (Using transformers for sentiment)
# ================================

def extract_student_text(transcript):
    """
    Extracts student lines.
    If a line explicitly starts with "Student:" (case-insensitive), uses that.
    Otherwise, any line not starting with "Salesperson:" or "Counselor:" is assumed to be student dialogue.
    """
    student_lines = []
    for line in transcript.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        lower_line = line_stripped.lower()
        if lower_line.startswith("student:"):
            student_lines.append(line_stripped.split("student:", 1)[1].strip())
        elif lower_line.startswith("salesperson:") or lower_line.startswith("counselor:"):
            continue
        else:
            student_lines.append(line_stripped)
    return "\n".join(student_lines)

def extract_salesperson_text(transcript):
    """
    Extracts counselor dialogue based on explicit labels or if the line contains counselor keywords.
    """
    counselor_keywords = [
        "university", "education", "programme", "mba", "fees", 
        "payment", "registration", "international", "linkedin", 
        "profile", "consultancy", "call", "whatsapp"
    ]
    counselor_lines = []
    for line in transcript.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        lower_line = line_stripped.lower()
        if lower_line.startswith("salesperson:") or lower_line.startswith("counselor:"):
            if lower_line.startswith("salesperson:"):
                counselor_lines.append(line_stripped.split("salesperson:", 1)[1].strip())
            else:
                counselor_lines.append(line_stripped.split("counselor:", 1)[1].strip())
        elif any(keyword in lower_line for keyword in counselor_keywords):
            counselor_lines.append(line_stripped)
    return "\n".join(counselor_lines)

def analyze_sentiment(text):
    """
    Analyzes sentiment using the Hugging Face sentiment pipeline on a sentence-by-sentence basis.
    Splits overly long sentences into chunks.
    Returns overall sentiment label, explanation, and average compound score.
    """
    sentences = nltk.sent_tokenize(text)
    compounds = []
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        if len(tokens) > 512:
            chunks = split_text_into_chunks(s, max_tokens=512)
            for chunk in chunks:
                result = sentiment_pipeline(chunk)[0]
                compound = result['score'] if result['label'] == "POSITIVE" else -result['score']
                compounds.append(compound)
        else:
            result = sentiment_pipeline(s)[0]
            compound = result['score'] if result['label'] == "POSITIVE" else -result['score']
            compounds.append(compound)
    avg_compound = sum(compounds) / len(compounds) if compounds else 0
    if avg_compound >= 0.2:
        sentiment = "Interested/Positive"
    elif avg_compound <= -0.2:
        sentiment = "Not Interested/Negative"
    else:
        sentiment = "Neutral/Uncertain"
    explanation = f"Average compound sentiment score is {avg_compound:.2f} based on sentence-level analysis."
    return sentiment, explanation, avg_compound

def analyze_salesperson(text, scaling=RATING_SCALING):
    """
    Analyzes the counselor's dialogue using the sentiment pipeline on a sentence-by-sentence basis.
    Splits long sentences into chunks, averages the compound scores, and maps to a rating (0-10).
    """
    sentences = nltk.sent_tokenize(text)
    compounds = []
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        if len(tokens) > 512:
            chunks = split_text_into_chunks(s, max_tokens=512)
            for chunk in chunks:
                result = sentiment_pipeline(chunk)[0]
                compound = result['score'] if result['label'] == "POSITIVE" else -result['score']
                compounds.append(compound)
        else:
            result = sentiment_pipeline(s)[0]
            compound = result['score'] if result['label'] == "POSITIVE" else -result['score']
            compounds.append(compound)
    avg_compound = sum(compounds) / len(compounds) if compounds else 0
    rating = (avg_compound + 1) * 5 * scaling
    return rating, abs(avg_compound)

# ================================
# AGGREGATED PERFORMANCE EVALUATION (Strict Mode, Aggregated)
# ================================
def evaluate_salesperson_performance_extended(text):
    """
    Provides aggregated feedback for the counselor based on the entire text.
    Instead of line-specific feedback, this function aggregates key metrics such as overall filler word usage,
    average sentence length, vocabulary diversity, and overall sentiment to produce a summary of strengths and areas to improve.
    """
    # Overall filler words count
    filler_words = ["like", "um", "uh", "you know"]
    total_filler = sum(text.lower().count(word) for word in filler_words)
    
    # Average sentence length
    sentences = nltk.sent_tokenize(text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    # Vocabulary diversity
    tokens = nltk.word_tokenize(text.lower())
    vocabulary_diversity = len(set(tokens)) / len(tokens) if tokens else 0
    
    # Overall sentiment for counselor text
    overall_sentiment, _, comp = analyze_sentiment(text)
    
    improvements = []
    praises = []
    
    if total_filler > 10:
        improvements.append(f"High overall filler word usage (total count: {total_filler}).")
    else:
        praises.append("Filler word usage is under control.")
    
    if avg_sentence_length > 20:
        improvements.append(f"Average sentence length is high ({avg_sentence_length:.1f} words).")
    else:
        praises.append("Sentences are generally concise.")
    
    if vocabulary_diversity < 0.3:
        improvements.append("Vocabulary diversity is low.")
    else:
        praises.append("Good vocabulary diversity.")
    
    if comp < -0.2:
        improvements.append("Overall negative sentiment detected. Tone may need improvement.")
    else:
        praises.append("Overall positive/professional tone.")
    
    improvements.append("Provide more personalized feedback to the candidate.")
    
    summary = ("Aggregated feedback indicates areas for improvement in clarity, sentence structure, "
               "and vocabulary usage. Enhancing these aspects will improve professional communication.")
    growth_areas = ["Reduce filler words", "Improve conciseness", "Enhance vocabulary", "Adopt a more positive tone"]
    
    return {
        "improvements": improvements,
        "praises": praises,
        "summary": summary,
        "growthAreas": growth_areas
    }

def generate_filler_words_chart(text):
    """
    Generates a bar chart for filler word frequencies.
    """
    filler_words = ["like", "um", "uh", "you know"]
    counts = {word: text.lower().count(word) for word in filler_words}
    fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                 labels={'x': 'Filler Word', 'y': 'Count'},
                 title="Filler Words Frequency")
    return fig

def suggest_courses(transcript):
    """
    Suggests courses for the student based on keywords in the transcript.
    """
    suggestions = []
    lower_t = transcript.lower()
    if "mba" in lower_t:
        suggestions.append("MBA Program")
    if "international" in lower_t:
        suggestions.append("International MBA Program")
    if "budget" in lower_t or "affordable" in lower_t:
        suggestions.append("Budget-Friendly MBA Program")
    return list(set(suggestions))

# ================================
# DIALOGUE TRANSCRIPT CREATION (Script Format)
# ================================
def create_dialogue(transcript):
    """
    Processes the full transcript line by line and labels each line with the speaker.
    Lines starting with "Student:" are labeled as "student - ".
    Lines starting with "Salesperson:" or "Counselor:" are labeled as "student counselor - ".
    Otherwise, if the line contains counselor keywords, it's labeled as "student counselor - ";
    else, as "student - ".
    Returns the dialogue in a script/play format.
    """
    counselor_keywords = [
        "university", "education", "programme", "mba", "fees", "payment",
        "registration", "international", "linkedin", "profile", "consultancy",
        "call", "whatsapp"
    ]
    script_lines = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if lower_line.startswith("student:"):
            script_lines.append("student - " + line.split("student:", 1)[1].strip())
        elif lower_line.startswith("salesperson:") or lower_line.startswith("counselor:"):
            script_lines.append("student counselor - " + line.split(":", 1)[1].strip())
        else:
            if any(keyword in lower_line for keyword in counselor_keywords):
                script_lines.append("student counselor - " + line)
            else:
                script_lines.append("student - " + line)
    return "\n".join(script_lines)

# ================================
# PROCESS TRANSCRIPT FUNCTION
# ================================
def process_transcript(transcript):
    """
    Processes the transcript:
      - Extracts student dialogue.
      - Extracts student counselor dialogue.
      - Analyzes student dialogue for overall sentiment.
      - Analyzes counselor dialogue for rating.
      - Evaluates counselor performance (aggregated feedback).
      - Suggests courses.
      - Creates a script-style dialogue transcript.
    Returns the results as a JSON-formatted string.
    """
    student_text = extract_student_text(transcript)
    sentiment, explanation, comp = analyze_sentiment(student_text)
    
    counselor_text = extract_salesperson_text(transcript)
    if counselor_text:
        rating, sp_compound = analyze_salesperson(counselor_text)
        aggregated_feedback = evaluate_salesperson_performance_extended(counselor_text)
    else:
        rating = None
        aggregated_feedback = {"improvements": [], "praises": [], "summary": "", "growthAreas": []}
    
    suggestions = suggest_courses(transcript)
    dialogue_transcript = create_dialogue(transcript)
    
    output = {
        "studentSentiment": sentiment,
        "explanation": explanation,
        "counselorRatingOutOf10": round(rating, 2) if rating is not None else "N/A",
        "thingsToImprove": aggregated_feedback["improvements"],
        "thingsYouDidGreat": aggregated_feedback["praises"],
        "feedbackSummary": aggregated_feedback["summary"],
        "growthAreas": aggregated_feedback["growthAreas"],
        "suggestions": suggestions,
        "studentTranscript": student_text,
        "counselorTranscript": counselor_text,
        "dialogueTranscript": dialogue_transcript
    }
    return json.dumps(output, indent=2)

# ================================
# AUDIO TRANSCRIPTION FUNCTION USING WHISPER
# ================================
def transcribe_with_whisper(file_obj):
    """
    Transcribes an uploaded WAV audio file using OpenAI's Whisper.
    The uploaded file (BytesIO) is saved temporarily and passed to Whisper.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(file_obj.getvalue())
        tmp_file.flush()
        temp_file_path = tmp_file.name

    model = whisper.load_model("base")
    result = model.transcribe(temp_file_path)
    os.remove(temp_file_path)
    return result["text"]

# ================================
# SENTIMENT BREAKDOWN FOR COUNSELOR TRANSCRIPT
# ================================
def compute_sentiment_breakdown(text):
    """
    Splits the counselor transcript into sentences, computes sentiment for each,
    and returns counts of positive and negative sentences.
    """
    sentences = nltk.sent_tokenize(text)
    pos_count = 0
    neg_count = 0
    for s in sentences:
        res = sentiment_pipeline(s)[0]
        if res['label'] == "POSITIVE":
            pos_count += 1
        else:
            neg_count += 1
    total = pos_count + neg_count
    if total == 0:
        return {"Positive": 0, "Negative": 0}
    return {"Positive": pos_count, "Negative": neg_count}

# ================================
# STREAMLIT APP UI
# ================================
st.set_page_config(page_title="Call Analysis Dashboard", layout="centered")
st.title("Call Analysis Dashboard")
st.write("Upload your call recording (WAV format) to get the full transcript.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.info("Transcribing audio using Whisper, please wait...")
    uploaded_file.seek(0)
    transcript = transcribe_with_whisper(uploaded_file)
    
    st.subheader("Full Transcript")
    st.text_area("Transcript", transcript, height=300)
    
    # Perform analysis in the background
    results_json = process_transcript(transcript)
    results = json.loads(results_json)
    
    st.subheader("Analysis Results")
    st.markdown(f"**Student Sentiment Analysis:** {results['studentSentiment']}")
    st.markdown(f"**Explanation:** {results['explanation']}")
    st.markdown(f"**Student Counselor Rating:** {results['counselorRatingOutOf10']} / 10")
    
    st.markdown("### Detailed Feedback for Student Counselor")
    st.markdown("#### Things to Improve")
    if results["thingsToImprove"]:
        for item in results["thingsToImprove"]:
            st.markdown(f"- {item}")
    else:
        st.markdown("None")
        
    st.markdown("#### Things You Did Great")
    if results["thingsYouDidGreat"]:
        for item in results["thingsYouDidGreat"]:
            st.markdown(f"- {item}")
    else:
        st.markdown("None")
        
    st.markdown("#### Feedback Summary")
    st.markdown(results["feedbackSummary"])
    
    st.markdown("#### Growth Areas")
    if results["growthAreas"]:
        for item in results["growthAreas"]:
            st.markdown(f"- {item}")
    else:
        st.markdown("None")
        
    st.markdown("### Course Suggestions (For Student)")
    if results["suggestions"]:
        for item in results["suggestions"]:
            st.markdown(f"- {item}")
    else:
        st.markdown("No suggestions available.")
    
    # Generate a pie chart for counselor sentiment breakdown (sentence-level)
    counselor_text = extract_salesperson_text(transcript)
    if counselor_text:
        breakdown = compute_sentiment_breakdown(counselor_text)
        labels = list(breakdown.keys())
        values = list(breakdown.values())
        pie_fig = px.pie(names=labels, values=values, title="Student Counselor Sentiment Breakdown (Sentence-level)")
        st.plotly_chart(pie_fig)
    
    # Generate a bar chart for filler words frequency in counselor dialogue
    if counselor_text:
        filler_fig = generate_filler_words_chart(counselor_text)
        st.plotly_chart(filler_fig)
