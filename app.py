# Import necessary libraries
import pyaudio
import wave
import tempfile
import os
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gtts import gTTS
import io
import base64
import speech_recognition as sr
from googletrans import Translator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import time
import json

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Constants
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi'
}

# Function to detect language (enhanced implementation)
def detect_language(text):
    if not text:
        return 'en'
    
    # Hinglish detection patterns - more comprehensive
    hinglish_pattern = r'[a-zA-Z]+\s+[^\x00-\x7F]|[^\x00-\x7F]+\s+[a-zA-Z]'
    roman_hindi_words = ['hai', 'nahi', 'kya', 'aap', 'main', 'hum', 'tum', 'yeh', 'woh', 'kaise', 'kyun', 'kahan']
    
    # Check for romanized Hindi (Hinglish)
    words = word_tokenize(text.lower())
    hindi_word_count = sum(1 for word in words if word in roman_hindi_words)
    if hindi_word_count > 0 and len(words) > 0 and hindi_word_count / len(words) > 0.2:
        return 'hinglish'
    
    if re.search(hinglish_pattern, text):
        return 'hinglish'
    
    # Script-based language detection
    patterns = {
        'hi': r'[\u0900-\u097F]',  # Hindi
        'mr': r'[\u0900-\u097F]',  # Marathi (uses Devanagari like Hindi)
        'gu': r'[\u0A80-\u0AFF]',  # Gujarati
        'ta': r'[\u0B80-\u0BFF]',  # Tamil
        'te': r'[\u0C00-\u0C7F]',  # Telugu
        'bn': r'[\u0980-\u09FF]',  # Bengali
        'kn': r'[\u0C80-\u0CFF]',  # Kannada
        'ml': r'[\u0D00-\u0D7F]',  # Malayalam
        'pa': r'[\u0A00-\u0A7F]',  # Punjabi (Gurmukhi)
    }
    
    # Count character occurrences for each script
    lang_counts = {lang: len(re.findall(pattern, text)) for lang, pattern in patterns.items()}
    
    # Find the language with the most script characters
    max_lang = max(lang_counts.items(), key=lambda x: x[1]) if lang_counts else ('en', 0)
    
    # If we found significant script characters, return that language
    if max_lang[1] > 5:
        if max_lang[0] in ['hi', 'mr']:
            # Differentiate between Hindi and Marathi
            # This is simplified - in production use better linguistic markers
            marathi_markers = ['आहे', 'होते', 'आहेत', 'माझ', 'तुमच', 'त्याच', 'मराठी']
            if any(marker in text for marker in marathi_markers):
                return 'mr'
            return 'hi'
        return max_lang[0]
    
    # Default to English
    return 'en'

# Enhanced Translator class using Google Translate API
# Replace the SchemeTranslator class with this improved implementation
class SchemeTranslator:
    def __init__(self):
        # Use deep_translator instead of googletrans for more reliable translations
        # No need to initialize a translator object here
        self.translation_cache = {}  # Cache to improve performance
        
    def translate(self, text, target_lang):
        """Translate text to the target language"""
        from deep_translator import GoogleTranslator
        
        if not text or target_lang == 'en':
            return text
            
        # Check cache first
        cache_key = f"{text[:100]}|{target_lang}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
            
        try:
            # Handle long text by chunking (Google has a character limit)
            if len(text) > 5000:
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                translated_chunks = []
                for chunk in chunks:
                    # Use GoogleTranslator from deep_translator
                    translator = GoogleTranslator(source='auto', target=target_lang)
                    result = translator.translate(chunk)
                    translated_chunks.append(result)
                translated_text = ' '.join(translated_chunks)
            else:
                # For shorter text, translate in one go
                translator = GoogleTranslator(source='auto', target=target_lang)
                translated_text = translator.translate(text)
                
            # Cache the result
            self.translation_cache[cache_key] = translated_text
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            # Fallback to original text
            return text
            
    def translate_to_english(self, text, source_lang=None):
        """Translate text to English"""
        from deep_translator import GoogleTranslator
        
        if not text:
            return text
            
        if source_lang == 'en' or detect_language(text) == 'en':
            return text
            
        # Check cache first
        cache_key = f"{text[:100]}|en"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
            
        try:
            # Split long text into chunks
            if len(text) > 5000:
                chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
                translated_chunks = []
                for chunk in chunks:
                    # Use GoogleTranslator from deep_translator
                    translator = GoogleTranslator(source='auto', target='en')
                    result = translator.translate(chunk)
                    translated_chunks.append(result)
                translated_text = ' '.join(translated_chunks)
            else:
                # For shorter text, translate in one go
                translator = GoogleTranslator(source='auto', target='en')
                translated_text = translator.translate(text)
                
            # Cache the result
            self.translation_cache[cache_key] = translated_text
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            # Fallback to original text
            return text

# Replace the text_to_speech method in the SpeechProcessor class
def text_to_speech(self, text, lang='en'):
    """Convert text to speech in the specified language using pyttsx3 with fallback
    to gTTS for better multilingual support"""
    import pyttsx3
    import tempfile
    
    try:
        # Clean text for TTS - remove markdown and other formatting
        clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markdown
        clean_text = re.sub(r'\n\n', ' ', clean_text)  # Replace double newlines with space
        clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)  # Remove markdown links
        
        # For shorter text segments, try using pyttsx3 first (works offline)
        if len(clean_text) < 3000 and lang == 'en':
            try:
                # Use pyttsx3 for English (offline and better performance)
                engine = pyttsx3.init()
                # Create a temporary file to save the audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_mp3:
                    temp_filename = temp_mp3.name
                
                # Convert text to speech and save to file
                engine.save_to_file(clean_text, temp_filename)
                engine.runAndWait()
                
                # Read the audio file
                with open(temp_filename, 'rb') as f:
                    audio_bytes = f.read()
                
                # Clean up
                os.unlink(temp_filename)
                
                # Create HTML audio element
                audio_b64 = base64.b64encode(audio_bytes).decode()
                audio_html = f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                """
                return audio_html
            except Exception as e:
                print(f"pyttsx3 error: {e}")
                # Fall back to gTTS if pyttsx3 fails
                pass
                
        # For non-English or longer text, use gTTS (requires internet)
        # Split text if it's too long for TTS
        MAX_TTS_LENGTH = 3000
        text_parts = []
        
        if len(clean_text) > MAX_TTS_LENGTH:
            # Split by sentences or paragraphs
            chunks = re.split(r'(?<=[.!?])\s+', clean_text)
            current_chunk = ""
            
            for chunk in chunks:
                if len(current_chunk) + len(chunk) < MAX_TTS_LENGTH:
                    current_chunk += chunk + " "
                else:
                    if current_chunk:
                        text_parts.append(current_chunk.strip())
                    current_chunk = chunk + " "
            
            if current_chunk:
                text_parts.append(current_chunk.strip())
        else:
            text_parts = [clean_text]
            
        # Map language codes for gTTS if needed
        lang_map = {
            'hinglish': 'hi',  # Use Hindi for Hinglish
            'mr': 'hi',  # Use Hindi for Marathi
            'pa': 'pa',  # Punjabi
            'gu': 'gu',  # Gujarati
            'bn': 'bn',  # Bengali
            'ta': 'ta',  # Tamil
            'te': 'te',  # Telugu
            'kn': 'kn',  # Kannada
            'ml': 'ml'   # Malayalam
        }
        
        if lang in lang_map:
            lang = lang_map[lang]
            
        # Process each part
        audio_bytes_list = []
        for part in text_parts:
            # Generate speech
            try:
                # Use gTTS for better multilingual support
                from gtts import gTTS
                tts = gTTS(text=part, lang=lang, slow=False)
                
                # Save to BytesIO object
                part_audio_bytes = io.BytesIO()
                tts.write_to_fp(part_audio_bytes)
                part_audio_bytes.seek(0)
                audio_bytes_list.append(part_audio_bytes.read())
            except Exception as e:
                print(f"gTTS error for part: {e}")
                continue
        
        # Combine all audio parts
        combined_audio = b''.join(audio_bytes_list)
            
        # Create HTML audio element
        audio_b64 = base64.b64encode(combined_audio).decode()
        audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        """
        return audio_html
        
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        # Fallback to basic message if all TTS methods fail
        return f"<p>Text-to-speech error: {e}</p>"

# Enhanced Hinglish processor
class HinglishProcessor:
    def __init__(self):
        self.stop_words_en = set(stopwords.words('english'))
        self.stop_words_hi = set(['का', 'के', 'में', 'है', 'हैं', 'से', 'और', 'पर', 'को', 'की', 'कि'])
        
        # Load Hindi-English dictionary (simplified example - in production use a real dictionary)
        self.hindi_english_dict = {
            'kya': 'क्या',
            'hai': 'है',
            'nahi': 'नहीं',
            'aap': 'आप',
            'main': 'मैं',
            'hum': 'हम',
            'yeh': 'यह',
            'woh': 'वह',
            'scheme': 'योजना',
            'sarkari': 'सरकारी',
            'kaise': 'कैसे',
            'milega': 'मिलेगा',
            'chahiye': 'चाहिए',
            'kahan': 'कहां',
            'apply': 'आवेदन'
        }
        
    def process_hinglish(self, text):
        # More sophisticated Hinglish processing
        words = text.split()
        processed_words = []
        
        for word in words:
            # Check dictionary first
            if word.lower() in self.hindi_english_dict:
                processed_words.append(self.hindi_english_dict[word.lower()])
                continue
                
            # Skip stop words
            if word.lower() in self.stop_words_en or word in self.stop_words_hi:
                processed_words.append(word)
                continue
                
            # Try to transliterate
            if re.match(r'^[a-zA-Z]+$', word):
                try:
                    # Attempt to transliterate from Latin to Devanagari
                    transliterated = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
                    # If transliteration changed the word, use it
                    if transliterated != word:
                        processed_words.append(transliterated)
                    else:
                        processed_words.append(word)
                except:
                    processed_words.append(word)
            else:
                processed_words.append(word)
                
        return ' '.join(processed_words)

# Enhanced Scheme Retrieval System with caching
class SchemeRetrievalSystem:
    def __init__(self, data_path):
        try:
            self.df = pd.read_csv(data_path)
        except Exception:
            self.df = pd.DataFrame(columns=[
                'scheme_name','summary','eligibility_criteria',
                'application_process','documents_required','scheme_link','category'
            ])
        self.df.fillna('', inplace=True)
        # ensure category column exists
        if 'category' not in self.df.columns:
            self.df['category'] = 'General'
        # combine text for search
        tag_cols = [c for c in self.df.columns if c.startswith('tags/')]
        self.df['search_text'] = (
            self.df['scheme_name'] + ' ' + self.df['summary'] + ' ' +
            self.df['eligibility_criteria'] + ' ' + ' '.join(self.df.get(tag_cols, []).astype(str).apply(' '.join, axis=1))
        )
        self.df['search_text'] = self.df['search_text'].str.lower().str.replace(r"[^\w\s]", ' ', regex=True)
        # build TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
        self.query_cache = {}
            
    def process_data(self):
        # Fill NaN values with empty strings
        self.df = self.df.fillna('')
        
        # Create combined text field for search
        self.df['search_text'] = (
            self.df['scheme_name'] + ' ' + 
            self.df['summary'] + ' ' + 
            self.df['eligibility_criteria']
        )
        
        # Add tags if they exist in the dataframe
        if 'tags/0' in self.df.columns:
            tag_columns = [col for col in self.df.columns if col.startswith('tags/')]
            for col in tag_columns:
                self.df['search_text'] += ' ' + self.df[col]
        
        # Clean text
        self.df['search_text'] = self.df['search_text'].apply(
            lambda x: re.sub(r'[^\w\s]', ' ', x.lower())
        )
        
    def build_search_index(self):
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
        
    def search_schemes(self, query, top_n=5):
        key = f"{query}|{top_n}"
        if key in self.query_cache:
            return self.query_cache[key]
        qvec = self.vectorizer.transform([query.lower()])
        sim = cosine_similarity(qvec, self.tfidf_matrix).flatten()
        idxs = sim.argsort()[-top_n:][::-1]
        results = []
        for idx in idxs:
            if sim[idx] > 0.1:
                row = self.df.iloc[idx]
                info = {
                    'scheme_name': row['scheme_name'],
                    'summary': row['summary'],
                    'eligibility': row['eligibility_criteria'],
                    'application_process': row['application_process'],
                    'documents_required': row['documents_required'],
                    'scheme_link': row.get('scheme_link',''),
                    'category': row['category'],
                    'similarity_score': float(sim[idx])
                }
                results.append(info)
        self.query_cache[key] = results
        return results

# Enhanced NLP Processor for scheme queries
class SchemeQueryProcessor:
    def __init__(self):
        # Initialize NLP pipeline
        try:
            # Use a more task-specific model for intent classification
            self.query_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Entity extraction pipeline
            self.ner_pipeline = pipeline(
                "token-classification", 
                model="Jean-Baptiste/roberta-large-ner-english"
            )
            
            # For extracting topics/keywords from queries
            self.keywords_extractor = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading NLP models: {e}")
            self.model_loaded = False
        
        # For translating between languages
        self.translator = SchemeTranslator()
        
        # For Hinglish processing
        self.hinglish_processor = HinglishProcessor()
        
        # Intent recognition patterns
        self.intent_patterns = {
            "eligibility": [
                r"(eligible|eligibility|qualify|qualification|who can apply|am i eligible)",
                r"(who (is|are) eligible)",
                r"(can i apply|can i get)",
                r"(criteria|requirements for eligibility)"
            ],
            "application": [
                r"(how to apply|application process|steps to apply|apply online)",
                r"(where to apply|how (can|do) i apply)",
                r"(procedure|registration process)"
            ],
            "documents": [
                r"(documents|documentation|papers|certificates|proof)",
                r"(what (documents|papers|id) (do i need|are required|should i submit))",
                r"(required documents)"
            ],
            "benefits": [
                r"(benefits|advantage|what will i get|how much money)",
                r"(assistance|support|aid|help|subsidy|how much)",
                r"(amount|payment|financial support)"
            ],
            "deadline": [
                r"(deadline|last date|closing date|due date)",
                r"(when (is|was) the last date|time limit)"
            ]
        }
        
    def extract_intent_rule_based(self, query):
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default to general information
        return "general"
        
    def extract_scheme_types(self, query):
        # Categories of schemes to detect
        scheme_categories = [
            "education", "scholarship", "farming", "agriculture", "healthcare", "medical",
            "housing", "rural", "urban", "employment", "job", "startup", "business",
            "women", "girl", "female", "senior citizen", "elderly", "disability", "pension",
            "youth", "skill development", "training", "loan", "subsidy", "insurance"
        ]
        
        # Check if query mentions any scheme category
        query_lower = query.lower()
        mentioned_categories = [cat for cat in scheme_categories if cat in query_lower]
        
        return mentioned_categories
        
    def process_query(self, query, detected_lang):
        # Normalize the query based on language
        if detected_lang == 'hinglish':
            # Process Hinglish
            processed_query = self.hinglish_processor.process_hinglish(query)
            # Translate to English for processing
            english_query = self.translator.translate_to_english(processed_query)
        else:
            # Translate to English for processing
            english_query = self.translator.translate_to_english(query, detected_lang)
            
        # Classify query intent using both methods
        if self.model_loaded:
            model_intent = self.classify_query_type(english_query)
        else:
            model_intent = "general"
            
        # Rule-based intent extraction as backup
        rule_intent = self.extract_intent_rule_based(english_query)
        
        # Combine both methods, preferring rule-based for specific intents
        final_intent = rule_intent if rule_intent != "general" else model_intent
        
        # Extract scheme types/categories mentioned
        scheme_types = self.extract_scheme_types(english_query)
        
        return english_query, final_intent, scheme_types
        
    def classify_query_type(self, query):
        if not self.model_loaded:
            return "general"
            
        # Define the possible query types
        candidate_labels = [
            "eligibility", 
            "application", 
            "documents", 
            "benefits",
            "deadline",
            "general"
        ]
        
        # Classify the query
        result = self.query_classifier(query, candidate_labels)
        
        # Return the most likely query type
        return result['labels'][0]
    
    def format_scheme_response(self, scheme, query_type, scheme_types=None):
        # Start with scheme name
        response = f"**{scheme['scheme_name']}**\n\n"
        
        # Generate appropriate response based on query type
        if query_type == "eligibility":
            response += f"**Eligibility Criteria:**\n{scheme['eligibility']}"
            
        elif query_type == "application":
            response += f"**Application Process:**\n{scheme['application_process']}"
            
        elif query_type == "documents":
            response += f"**Documents Required:**\n{scheme['documents_required']}"
            
        elif query_type == "benefits":
            response += f"**Benefits & Summary:**\n{scheme['summary']}"
            
        elif query_type == "deadline":
            # Look for deadline information in text
            deadline_info = self.extract_deadline_info(scheme)
            if deadline_info:
                response += f"**Application Deadline:**\n{deadline_info}"
            else:
                response += "**Note:** Specific deadline information is not available. Please check the scheme website for the most current deadline.\n\n"
                response += f"**Summary:**\n{scheme['summary'][:200]}...\n\n"
        else:  # default - general information
            # Provide a concise summary for general queries
            response += f"**Summary:**\n{scheme['summary']}\n\n"
            response += f"**Eligibility Criteria:**\n{scheme['eligibility']}"
            
        # Add link to official website if available
        if scheme.get('scheme_link') and scheme['scheme_link'] != "":
            response += f"\n\n**More Information:** [Official Website]({scheme['scheme_link']})"
            
        return response
        
    def extract_deadline_info(self, scheme):
        # Look for deadline information in the scheme texts
        deadline_patterns = [
            r"deadline[s]?[\s\:]+([^\.\n]+)",
            r"last date[\s\:]+([^\.\n]+)",
            r"closing date[\s\:]+([^\.\n]+)",
            r"due date[\s\:]+([^\.\n]+)",
            r"apply before[\s\:]+([^\.\n]+)",
            r"apply by[\s\:]+([^\.\n]+)",
            r"apply till[\s\:]+([^\.\n]+)",
            r"apply until[\s\:]+([^\.\n]+)"
        ]
        
        # Search in all text fields
        all_text = scheme['summary'] + " " + scheme['eligibility'] + " " + scheme['application_process']
        
        for pattern in deadline_patterns:
            matches = re.search(pattern, all_text, re.IGNORECASE)
            if matches:
                return matches.group(1).strip()
                
        return None

# Enhanced Speech processor with better error handling
class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust the energy threshold for better speech recognition
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
    def record_audio(self, duration=5):
        """Record audio for the specified duration"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        # Try to initialize PyAudio
        try:
            p = pyaudio.PyAudio()
        except Exception as e:
            return None, f"Error initializing audio: {str(e)}"
        
        try:
            stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
            
            st.write("🎤 Recording... Please speak now.")
            progress_bar = st.progress(0)
            
            frames = []
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                # Update progress bar
                progress_bar.progress((i + 1) / int(RATE / CHUNK * duration))
            
            st.write("✅ Recording complete!")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save temporarily to a WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_filename = temp_wav.name
                
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Read the file content
            with open(temp_filename, 'rb') as f:
                audio_bytes = f.read()
                
            # Clean up
            os.unlink(temp_filename)
            
            return audio_bytes, None
            
        except Exception as e:
            if 'p' in locals():
                p.terminate()
            return None, f"Error recording audio: {str(e)}"
    
    def speech_to_text(self, audio_bytes):
        try:
            # Check if we received audio data
            if audio_bytes is None or len(audio_bytes) == 0:
                return "No audio data received."
                
            # Process the audio data
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
            
            # Try multiple language models if available
            try:
                # First try Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data)
            except:
                try:
                    # Fall back to Sphinx if Google fails
                    text = self.recognizer.recognize_sphinx(audio_data)
                except:
                    return "Sorry, I couldn't understand the audio. Please try speaking more clearly."
                    
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio. Please try speaking more clearly."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def text_to_speech(self, text, lang='en'):
        """Convert text to speech in the specified language with enhanced support for Indian languages"""
        try:
            # Clean text for TTS - remove markdown and other formatting
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold markdown
            clean_text = re.sub(r'\n\n', ' ', clean_text)  # Replace double newlines with space
            clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)  # Remove markdown links
            
            # Proper language code mapping for gTTS
            lang_map = {
                'hinglish': 'hi',  # Use Hindi for Hinglish
                'mr': 'hi',        # Use Hindi for Marathi
                'pa': 'pa',        # Punjabi
                'gu': 'gu',        # Gujarati
                'bn': 'bn',        # Bengali
                'ta': 'ta',        # Tamil
                'te': 'te',        # Telugu
                'kn': 'kn',        # Kannada
                'ml': 'ml'         # Malayalam
            }
            
            # Convert language code if needed
            tts_lang = lang_map.get(lang, lang)
            
            # Split text if it's too long for TTS to handle
            MAX_TTS_LENGTH = 3000
            text_parts = []
            
            if len(clean_text) > MAX_TTS_LENGTH:
                # Split by sentences or paragraphs
                chunks = re.split(r'(?<=[.!?])\s+', clean_text)
                current_chunk = ""
                
                for chunk in chunks:
                    if len(current_chunk) + len(chunk) < MAX_TTS_LENGTH:
                        current_chunk += chunk + " "
                    else:
                        if current_chunk:
                            text_parts.append(current_chunk.strip())
                        current_chunk = chunk + " "
                
                if current_chunk:
                    text_parts.append(current_chunk.strip())
            else:
                text_parts = [clean_text]
            
            # Process each part
            audio_bytes_list = []
            
            # First attempt: Use gTTS for all languages (most reliable for multiple languages)
            try:
                for part in text_parts:
                    tts = gTTS(text=part, lang=tts_lang, slow=False)
                    
                    # Save to BytesIO object
                    part_audio_bytes = io.BytesIO()
                    tts.write_to_fp(part_audio_bytes)
                    part_audio_bytes.seek(0)
                    audio_bytes_list.append(part_audio_bytes.read())
                
                # If we got here, gTTS worked successfully
                combined_audio = b''.join(audio_bytes_list)
                
                # Create HTML audio element
                audio_b64 = base64.b64encode(combined_audio).decode()
                audio_html = f"""
                    <audio controls autoplay>
                        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                """
                return audio_html
                
            except Exception as e:
                # If gTTS fails, try pyttsx3 as fallback for English only
                print(f"gTTS error: {e}, trying pyttsx3 fallback for English...")
                if tts_lang == 'en':
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_mp3:
                            temp_filename = temp_mp3.name
                        
                        # Convert to speech and save
                        engine.save_to_file(clean_text, temp_filename)
                        engine.runAndWait()
                        
                        # Read the audio file
                        with open(temp_filename, 'rb') as f:
                            audio_bytes = f.read()
                        
                        # Clean up
                        os.unlink(temp_filename)
                        
                        # Create HTML audio element
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        audio_html = f"""
                            <audio controls autoplay>
                                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                        """
                        return audio_html
                        
                    except Exception as e2:
                        print(f"pyttsx3 fallback error: {e2}")
                        # Both TTS methods failed
                        return f"<p>Unable to generate speech in {SUPPORTED_LANGUAGES.get(lang, lang)}. Please check your internet connection.</p>"
                else:
                    # For non-English when gTTS fails
                    return f"<p>Unable to generate speech in {SUPPORTED_LANGUAGES.get(lang, lang)}. Please check your internet connection.</p>"
                    
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return f"<p>Text-to-speech error: {e}</p>"

# Enhanced Government Scheme Chatbot
class GovernmentSchemeChatbot:
    def __init__(self, data_path):
        self.scheme_retrieval = SchemeRetrievalSystem(data_path)
        self.query_processor = SchemeQueryProcessor()
        self.translator = SchemeTranslator()
        self.speech_processor = SpeechProcessor()
        self.context = {'history': []}

        # Keep conversation context
        self.context = {
            "last_schemes": [],
            "last_query_type": None,
            "user_preferences": {},
            "conversation_history": []
        }
        
    def process_input(self, user_input, input_type='text'):
        if not user_input:
            return "Please ask a question about government schemes.", None, 'en'
            
        if input_type == 'voice':
            # Process voice input
            if isinstance(user_input, tuple) and len(user_input) == 2:
                audio_bytes, error = user_input
                if error:
                    return error, None, 'en'
                user_input = self.speech_processor.speech_to_text(audio_bytes)
            else:
                user_input = self.speech_processor.speech_to_text(user_input)
                
            if user_input.startswith("Sorry") or user_input.startswith("An error"):
                return user_input, None, 'en'
        
        # Add to conversation history
        self.context["conversation_history"].append({"role": "user", "content": user_input})
        
        # Detect language
        detected_lang = detect_language(user_input)
        
        # Process query based on language
        english_query, query_type, scheme_types = self.query_processor.process_query(user_input, detected_lang)
        
        # Update context
        self.context["last_query_type"] = query_type
        
        # Search for relevant schemes
        schemes = self.scheme_retrieval.search_schemes(english_query)
        
        # Update context with found schemes
        self.context["last_schemes"] = schemes
        
        if not schemes:
            response = "I couldn't find any relevant government schemes. Could you please provide more details about what you're looking for? For example, mention if you're interested in education, healthcare, agriculture, or other specific areas."
        else:
            # Check if the query is about a specific scheme mentioned earlier
            if len(self.context["conversation_history"]) > 2:
                # Look for references to previously mentioned schemes in the current query
                for scheme in self.context["last_schemes"]:
                    scheme_name_lower = scheme['scheme_name'].lower()
                    if scheme_name_lower in english_query.lower():
                        # Format response for the specifically mentioned scheme
                        response = self.query_processor.format_scheme_response(scheme, query_type, scheme_types)
                        
                        # Add to conversation history
                        self.context["conversation_history"].append({"role": "assistant", "content": response})
                        
                        # Translate response back to user's language if needed
                        if detected_lang != 'en' and detected_lang != 'hinglish':
                            response = self.translator.translate(response, detected_lang)
                        
                        return response, schemes, detected_lang
            
            # Format response for the top scheme
            response = self.query_processor.format_scheme_response(schemes[0], query_type, scheme_types)
            
            # Add information about other schemes if available
            if len(schemes) > 1:
                response += "\n\n**Other Related Schemes:**\n"
                for i, scheme in enumerate(schemes[1:3], 1):  # Show 2 more related schemes
                    response += f"{i}. {scheme['scheme_name']}\n"
        
        # Add to conversation history
        self.context["conversation_history"].append({"role": "assistant", "content": response})
        
        # Translate response back to user's language if needed
        if detected_lang != 'en' and detected_lang != 'hinglish':
            response = self.translator.translate(response, detected_lang)
            
        # For Hinglish, we keep it in English as translating to Hinglish is complex
        # In a production system, you might want a dedicated Hinglish translator
            
        return response, schemes, detected_lang
        
    def get_audio_response(self, text, lang):
        return self.speech_processor.text_to_speech(text, lang)
    
    def get_followup_questions(self, schemes, query_type):
        """Generate follow-up questions based on context"""
        if not schemes:
            return []
            
        followup_questions = []
        
        # Generate different follow-up questions based on the current query type
        if query_type == "general":
            followup_questions.append(f"What are the eligibility criteria for {schemes[0]['scheme_name']}?")
            followup_questions.append(f"How can I apply for {schemes[0]['scheme_name']}?")
            
        elif query_type == "eligibility":
            followup_questions.append(f"What documents are required for {schemes[0]['scheme_name']}?")
            followup_questions.append(f"How can I apply for {schemes[0]['scheme_name']}?")
            
        elif query_type == "application":
            followup_questions.append(f"What documents are required for {schemes[0]['scheme_name']}?")
            followup_questions.append(f"What are the benefits of {schemes[0]['scheme_name']}?")
            
        elif query_type == "documents":
            followup_questions.append(f"What is the application process for {schemes[0]['scheme_name']}?")
            followup_questions.append(f"Am I eligible for {schemes[0]['scheme_name']}?")
            
        return followup_questions[:2]  # Return top 2 followup questions

# Streamlit UI
def main():
    st.set_page_config(page_title="Government Scheme Chatbot", page_icon="🤖", layout="wide")
    # session init
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = GovernmentSchemeChatbot("myschemes_scraped_combined.csv")
        st.session_state.data_loaded = True

    st.title("🇮🇳 Indian Government Schemes Assistant")
    st.markdown("Ask about any government scheme in your preferred language.")

    # derive category list dynamically
    df = st.session_state.chatbot.scheme_retrieval.df
    categories = ['All Categories'] + sorted(df['category'].dropna().unique().tolist())

    col1, col2 = st.columns([3,1])
    with st.sidebar:
        st.header("Settings")
        selected_lang = st.selectbox("Output Language", options=SUPPORTED_LANGUAGES.keys(), format_func=lambda k: SUPPORTED_LANGUAGES[k])
        input_method = st.radio("Input Method", ["Text","Voice"])
        st.header("Filter by Category")
        selected_category = st.selectbox("Category", options=categories)
        st.markdown("---")
        st.header("Examples")
        st.markdown(
            "- What schemes are available for farmers?\n"
            "- List schemes for Healthcare\n"
            "- Am I eligible for PM Kisan?"
        )

    if not st.session_state.data_loaded:
        st.error("Failed to load schemes data.")
        return

    with col1:
        st.subheader("Your Query")
        if input_method == 'Text':
            user_input = st.text_input("Type your question:", key='user_query')
            if st.button("Submit") and user_input:
                with st.spinner("Processing..."):
                    # special case: list schemes by category
                    if user_input.lower().startswith('list') and 'scheme' in user_input.lower():
                        if selected_category == 'All Categories':
                            display = "Please select a specific category to list its schemes."  
                        else:
                            names = df[df['category']==selected_category]['scheme_name'].tolist()
                            if names:
                                display = f"**Schemes under {selected_category}:**\n" + '\n'.join([f"{i+1}. {n}" for i,n in enumerate(names)])
                            else:
                                display = f"No schemes found under {selected_category}."
                    else:
                        # general case
                        response, schemes, lang = st.session_state.chatbot.process_input(user_input)
                        # filter schemes by category
                        schemes = [s for s in schemes if selected_category=='All Categories' or s['category']==selected_category]
                        display = response
                    # translate
                    if selected_lang != 'en':
                        display = st.session_state.chatbot.translator.translate(display, selected_lang)
                    st.markdown(display)
                    # save history
                    st.session_state.chat_history.append({'user':user_input,'bot':display,'lang':selected_lang})
                    # speak
                    audio = st.session_state.chatbot.get_audio_response(display, selected_lang)
                    if audio:
                        st.markdown(audio, unsafe_allow_html=True)
        else:
            st.write("Click to record:")
            duration = st.slider("Seconds to record",3,15,5)
            if st.button("Record"):
                audio_bytes, err = st.session_state.chatbot.speech_processor.record_audio(duration)
                if err:
                    st.error(err)
                else:
                    with st.spinner("Recognizing..."):
                        text = st.session_state.chatbot.speech_processor.speech_to_text(audio_bytes)
                        st.info(f"Heard: {text}")
                        if text and not text.lower().startswith('sorry'):
                            if text.lower().startswith('list') and 'scheme' in text.lower():
                                # same special-case
                                if selected_category=='All Categories':
                                    display = "Please select a category to list its schemes."
                                else:
                                    names = df[df['category']==selected_category]['scheme_name'].tolist()
                                    display = f"**Schemes under {selected_category}:**\n" + '\n'.join([f"{i+1}. {n}" for i,n in enumerate(names)]) if names else f"No schemes under {selected_category}."
                            else:
                                response, schemes, lang = st.session_state.chatbot.process_input(text)
                                schemes = [s for s in schemes if selected_category=='All Categories' or s['category']==selected_category]
                                display = response
                            if selected_lang!='en':
                                display = st.session_state.chatbot.translator.translate(display, selected_lang)
                            st.markdown(display)
                            st.session_state.chat_history.append({'user':text,'bot':display,'lang':selected_lang})
                            audio_html = st.session_state.chatbot.get_audio_response(display, selected_lang)
                            if audio_html:
                                st.markdown(audio_html, unsafe_allow_html=True)

    st.subheader("Conversation History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['bot']}")
        if st.button(f"🔊 Listen", key=f"listen_{i}"):
            html = st.session_state.chatbot.get_audio_response(chat['bot'], chat['lang'])
            if html:
                st.markdown(html, unsafe_allow_html=True)
        st.markdown("---")

if __name__ == '__main__':
    main()

