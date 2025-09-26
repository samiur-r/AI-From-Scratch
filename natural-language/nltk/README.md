# NLTK Quick Reference

NLTK (Natural Language Toolkit) is a comprehensive platform for building Python programs to work with human language data, providing easy-to-use interfaces to over 50 corpora and lexical resources along with text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

### Installation
```bash
# Install NLTK
pip install nltk

# Additional dependencies for specific features
pip install matplotlib numpy scipy

# For advanced features
pip install scikit-learn pandas
```

### Importing NLTK
```python
import nltk
import string
from collections import Counter
import matplotlib.pyplot as plt

# Download required data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
# nltk.download('brown')
```

* * * * *

## 1. Text Tokenization
```python
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

# Sample text
text = "Hello world! This is NLTK. It's great for NLP tasks."

# Word tokenization
words = word_tokenize(text)
print("Words:", words)
# Output: ['Hello', 'world', '!', 'This', 'is', 'NLTK', '.', 'It', "'s", 'great', 'for', 'NLP', 'tasks', '.']

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
# Output: ['Hello world!', 'This is NLTK.', "It's great for NLP tasks."]

# Custom tokenization (only alphabetic tokens)
tokenizer = RegexpTokenizer(r'\w+')
alpha_words = tokenizer.tokenize(text)
print("Alpha words:", alpha_words)
# Output: ['Hello', 'world', 'This', 'is', 'NLTK', 'It', 's', 'great', 'for', 'NLP', 'tasks']

# Tokenize multiple sentences
paragraph = """
Natural language processing is fascinating.
It involves computational linguistics.
NLTK makes it accessible to everyone!
"""

sentences = sent_tokenize(paragraph)
all_words = []
for sentence in sentences:
    words = word_tokenize(sentence.lower())
    all_words.extend(words)

print(f"Total words: {len(all_words)}")
print(f"Unique words: {len(set(all_words))}")
```

## 2. Text Cleaning and Preprocessing
```python
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Comprehensive text cleaning function"""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Remove short words (less than 3 characters)
    words = [word for word in words if len(word) >= 3]

    return words

# Example usage
sample_text = "The running dogs were quickly jumping over the fences! Amazing, isn't it?"

# Clean text
cleaned_words = clean_text(sample_text)
print("Cleaned words:", cleaned_words)

# Apply stemming
stemmed_words = [stemmer.stem(word) for word in cleaned_words]
print("Stemmed words:", stemmed_words)

# Apply lemmatization (better than stemming)
lemmatized_words = [lemmatizer.lemmatize(word) for word in cleaned_words]
print("Lemmatized words:", lemmatized_words)

# Complete preprocessing pipeline
def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """Complete text preprocessing pipeline"""

    # Clean text
    words = clean_text(text)

    # Apply stemming or lemmatization
    if use_stemming:
        words = [stemmer.stem(word) for word in words]
    elif use_lemmatization:
        words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Example
original = "The cats are running quickly through the beautiful gardens."
processed = preprocess_text(original)
print(f"Original: {original}")
print(f"Processed: {processed}")
```

## 3. Part-of-Speech Tagging
```python
from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# POS tagging
text = "The quick brown fox jumps over the lazy dog."
words = word_tokenize(text)
pos_tags = pos_tag(words)

print("POS Tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")

# Common POS tag meanings
pos_meanings = {
    'NN': 'Noun, singular',
    'NNS': 'Noun, plural',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, present participle',
    'JJ': 'Adjective',
    'DT': 'Determiner',
    'IN': 'Preposition',
    'RB': 'Adverb'
}

# Extract specific parts of speech
def extract_pos(text, target_pos=['NN', 'NNS', 'JJ']):
    """Extract words with specific POS tags"""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    extracted = {}
    for pos in target_pos:
        extracted[pos] = [word for word, tag in pos_tags if tag == pos]

    return extracted

# Example
sample = "The beautiful flowers bloom magnificently in the garden."
extracted_pos = extract_pos(sample)
print("\nExtracted POS:")
for pos, words in extracted_pos.items():
    print(f"{pos}: {words}")

# Named Entity Recognition
text_with_entities = "Barack Obama was born in Hawaii. He served as President of the United States."
words = word_tokenize(text_with_entities)
pos_tags = pos_tag(words)
named_entities = ne_chunk(pos_tags)

print("\nNamed Entities:")
for chunk in named_entities:
    if isinstance(chunk, Tree):
        entity = " ".join([token for token, pos in chunk.leaves()])
        label = chunk.label()
        print(f"{entity}: {label}")
```

## 4. Frequency Analysis and N-grams
```python
from nltk import FreqDist, bigrams, trigrams
from nltk.util import ngrams

# Sample text for analysis
corpus = """
Natural language processing is a fascinating field of artificial intelligence.
It involves the interaction between computers and human language.
Natural language processing has many applications in modern technology.
Machine learning techniques are often used in natural language processing.
"""

# Preprocess the corpus
words = preprocess_text(corpus)

# Frequency distribution
freq_dist = FreqDist(words)

print("Most common words:")
for word, freq in freq_dist.most_common(10):
    print(f"{word}: {freq}")

# Plot frequency distribution
plt.figure(figsize=(12, 6))
freq_dist.plot(20, title="Word Frequency Distribution")
plt.show()

# N-grams analysis
def analyze_ngrams(text, n=2, top_k=10):
    """Analyze n-grams in text"""
    words = preprocess_text(text)

    # Generate n-grams
    n_grams = list(ngrams(words, n))

    # Count frequency
    ngram_freq = FreqDist(n_grams)

    print(f"Top {top_k} {n}-grams:")
    for ngram, freq in ngram_freq.most_common(top_k):
        ngram_str = " ".join(ngram)
        print(f"'{ngram_str}': {freq}")

    return ngram_freq

# Analyze bigrams and trigrams
print("Bigram Analysis:")
bigram_freq = analyze_ngrams(corpus, n=2, top_k=5)

print("\nTrigram Analysis:")
trigram_freq = analyze_ngrams(corpus, n=3, top_k=5)

# Character-level analysis
def character_frequency(text):
    """Analyze character frequency"""
    # Remove spaces and convert to lowercase
    chars = [char.lower() for char in text if char.isalpha()]
    char_freq = FreqDist(chars)

    print("Character frequency:")
    for char, freq in char_freq.most_common(10):
        print(f"'{char}': {freq}")

    return char_freq

char_freq = character_frequency(corpus)
```

## 5. Sentiment Analysis
```python
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    # Get sentiment scores
    scores = sia.polarity_scores(text)

    # Determine overall sentiment
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return {
        'sentiment': sentiment,
        'scores': scores,
        'confidence': abs(scores['compound'])
    }

# Example sentences
sentences = [
    "I love this movie! It's absolutely fantastic.",
    "This product is terrible. I hate it.",
    "The weather is okay today.",
    "NLTK is an amazing library for NLP tasks!",
    "I'm not sure how I feel about this."
]

print("Sentiment Analysis Results:")
print("-" * 50)
for sentence in sentences:
    result = analyze_sentiment(sentence)
    print(f"Text: {sentence}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Scores: {result['scores']}")
    print()

# Analyze sentiment of multiple texts
def batch_sentiment_analysis(texts):
    """Analyze sentiment for multiple texts"""
    results = []

    for text in texts:
        result = analyze_sentiment(text)
        results.append({
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence']
        })

    return results

# Example with reviews
reviews = [
    "The service was excellent and the food was delicious!",
    "Poor quality product, would not recommend.",
    "Average experience, nothing special.",
    "Outstanding customer support, very helpful staff."
]

batch_results = batch_sentiment_analysis(reviews)

# Summarize results
sentiment_counts = {}
for result in batch_results:
    sentiment = result['sentiment']
    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

print("Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count}")
```

## 6. Text Classification
```python
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load movie reviews dataset (if available)
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]
# random.shuffle(documents)

# Alternative: Create sample data for demonstration
def create_sample_data():
    """Create sample text classification data"""
    positive_texts = [
        "This movie is absolutely fantastic and amazing",
        "I love this film it's wonderful and great",
        "Excellent story and outstanding acting performance",
        "Beautiful cinematography and brilliant direction",
        "Awesome movie with incredible special effects"
    ]

    negative_texts = [
        "This movie is terrible and boring",
        "Awful film with poor acting and bad story",
        "Horrible movie I hated every minute",
        "Terrible plot and disappointing ending",
        "Worst movie ever made completely unwatchable"
    ]

    data = []
    labels = []

    for text in positive_texts:
        data.append(text)
        labels.append('positive')

    for text in negative_texts:
        data.append(text)
        labels.append('negative')

    return data, labels

# Create and prepare data
texts, labels = create_sample_data()

# Text preprocessing for classification
def preprocess_for_classification(text):
    """Preprocess text for classification"""
    # Clean and tokenize
    words = clean_text(text)
    # Join back to string for vectorizer
    return " ".join(words)

# Preprocess all texts
processed_texts = [preprocess_for_classification(text) for text in texts]

# Create classification pipeline
classifier_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    processed_texts, labels, test_size=0.3, random_state=42
)

# Train classifier
classifier_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = classifier_pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test with new examples
new_texts = [
    "This is an incredible movie with amazing acting",
    "Boring film with terrible plot and bad acting"
]

for text in new_texts:
    processed = preprocess_for_classification(text)
    prediction = classifier_pipeline.predict([processed])[0]
    probability = max(classifier_pipeline.predict_proba([processed])[0])

    print(f"Text: {text}")
    print(f"Predicted: {prediction} (confidence: {probability:.3f})")
    print()
```

## 7. Text Similarity and Distance
```python
from nltk.metrics import jaccard_distance, edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_text_similarities(text1, text2):
    """Calculate various similarity metrics between two texts"""

    # Preprocess texts
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))

    # Jaccard similarity
    jaccard_sim = 1 - jaccard_distance(words1, words2)

    # Edit distance (Levenshtein)
    edit_dist = edit_distance(text1.lower(), text2.lower())

    # Cosine similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return {
        'jaccard_similarity': jaccard_sim,
        'edit_distance': edit_dist,
        'cosine_similarity': cosine_sim
    }

# Example texts
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A fast brown fox leaps over a sleepy dog"
text3 = "Python is a great programming language"

# Calculate similarities
print("Text Similarity Analysis:")
print("-" * 40)

similarities_1_2 = calculate_text_similarities(text1, text2)
print(f"Text 1 vs Text 2:")
for metric, value in similarities_1_2.items():
    print(f"  {metric}: {value:.3f}")

similarities_1_3 = calculate_text_similarities(text1, text3)
print(f"\nText 1 vs Text 3:")
for metric, value in similarities_1_3.items():
    print(f"  {metric}: {value:.3f}")

# Document similarity matrix
def create_similarity_matrix(documents):
    """Create similarity matrix for multiple documents"""
    n = len(documents)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                sim = calculate_text_similarities(documents[i], documents[j])
                similarity_matrix[i][j] = sim['cosine_similarity']
            else:
                similarity_matrix[i][j] = 1.0

    return similarity_matrix

# Example with multiple documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing deals with human language",
    "AI and ML are transforming various industries"
]

sim_matrix = create_similarity_matrix(documents)
print(f"\nDocument Similarity Matrix:")
print(sim_matrix)

# Find most similar documents
max_sim = 0
most_similar = (0, 0)
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if sim_matrix[i][j] > max_sim:
            max_sim = sim_matrix[i][j]
            most_similar = (i, j)

print(f"\nMost similar documents (similarity: {max_sim:.3f}):")
print(f"Doc {most_similar[0]}: {documents[most_similar[0]]}")
print(f"Doc {most_similar[1]}: {documents[most_similar[1]]}")
```

## 8. Corpus Analysis and Information Extraction
```python
from nltk.corpus import brown
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

# Working with corpora
def analyze_corpus(corpus_words, sample_size=1000):
    """Analyze a corpus for various linguistic features"""

    # Take a sample if corpus is large
    if len(corpus_words) > sample_size:
        corpus_sample = corpus_words[:sample_size]
    else:
        corpus_sample = corpus_words

    # Basic statistics
    total_words = len(corpus_sample)
    unique_words = len(set(corpus_sample))

    print(f"Corpus Analysis:")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Lexical diversity: {unique_words/total_words:.3f}")

    # Frequency analysis
    freq_dist = FreqDist(corpus_sample)
    print(f"\nMost common words:")
    for word, freq in freq_dist.most_common(10):
        print(f"  {word}: {freq}")

    # Average word length
    avg_word_length = np.mean([len(word) for word in corpus_sample])
    print(f"\nAverage word length: {avg_word_length:.2f}")

    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'lexical_diversity': unique_words/total_words,
        'avg_word_length': avg_word_length,
        'freq_dist': freq_dist
    }

# Example with custom corpus
sample_corpus = """
Artificial intelligence is revolutionizing the way we work and live.
Machine learning algorithms can analyze vast amounts of data.
Natural language processing enables computers to understand human language.
Deep learning networks are inspired by the human brain.
These technologies are transforming industries across the globe.
"""

# Tokenize and clean
corpus_words = preprocess_text(sample_corpus)
analysis_result = analyze_corpus(corpus_words)

# Collocation analysis
def find_collocations(text, num_collocations=5):
    """Find meaningful word collocations"""
    words = preprocess_text(text)

    # Bigram collocations
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigram_finder.apply_freq_filter(2)  # Only bigrams that appear 2+ times

    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, num_collocations)

    # Trigram collocations
    trigram_finder = TrigramCollocationFinder.from_words(words)
    trigram_finder.apply_freq_filter(2)

    trigrams = trigram_finder.nbest(TrigramAssocMeasures.likelihood_ratio, num_collocations)

    print(f"\nTop {num_collocations} Bigram Collocations:")
    for bigram in bigrams:
        print(f"  {' '.join(bigram)}")

    print(f"\nTop {num_collocations} Trigram Collocations:")
    for trigram in trigrams:
        print(f"  {' '.join(trigram)}")

    return bigrams, trigrams

# Find collocations
bigrams, trigrams = find_collocations(sample_corpus, num_collocations=3)

# Keyword extraction using frequency and POS
def extract_keywords(text, num_keywords=10):
    """Extract keywords from text using frequency and POS filtering"""

    # Tokenize and get POS tags
    words = word_tokenize(text.lower())
    pos_tags = pos_tag(words)

    # Filter for nouns, adjectives, and verbs
    keyword_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    keywords = [word for word, pos in pos_tags if pos in keyword_pos and len(word) > 3]

    # Remove stopwords
    keywords = [word for word in keywords if word not in stop_words]

    # Calculate frequency
    keyword_freq = FreqDist(keywords)

    print(f"\nTop {num_keywords} Keywords:")
    for word, freq in keyword_freq.most_common(num_keywords):
        print(f"  {word}: {freq}")

    return keyword_freq.most_common(num_keywords)

# Extract keywords
keywords = extract_keywords(sample_corpus, num_keywords=8)
```

## 9. Language Detection and Text Generation
```python
import random

# Simple language detection using character frequency
def simple_language_detection(text, languages=['english', 'spanish', 'french']):
    """Simple language detection based on common words"""

    # Common words for different languages
    language_words = {
        'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
        'spanish': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
        'french': ['le', 'de', 'et', 'un', 'la', 'que', 'est', 'pour', 'dans', 'ce']
    }

    # Tokenize and lowercase
    words = word_tokenize(text.lower())

    # Count matches for each language
    language_scores = {}
    for lang in languages:
        if lang in language_words:
            score = sum(1 for word in words if word in language_words[lang])
            language_scores[lang] = score

    # Return most likely language
    detected_language = max(language_scores, key=language_scores.get)
    confidence = language_scores[detected_language] / len(words) if words else 0

    return detected_language, confidence, language_scores

# Test language detection
english_text = "The quick brown fox jumps over the lazy dog in the forest."
spanish_text = "El rápido zorro marrón salta sobre el perro perezoso en el bosque."

print("Language Detection:")
lang, conf, scores = simple_language_detection(english_text)
print(f"Text: {english_text}")
print(f"Detected: {lang} (confidence: {conf:.3f})")
print(f"Scores: {scores}")

lang, conf, scores = simple_language_detection(spanish_text)
print(f"\nText: {spanish_text}")
print(f"Detected: {lang} (confidence: {conf:.3f})")
print(f"Scores: {scores}")

# Simple text generation using n-grams
def generate_text_ngrams(corpus, n=2, length=10):
    """Generate text using n-gram model"""

    # Preprocess corpus
    words = preprocess_text(corpus)

    if len(words) < n:
        return "Corpus too small for n-gram generation"

    # Create n-grams
    ngrams_list = list(ngrams(words, n))

    # Create transition dictionary
    transitions = {}
    for ngram in ngrams_list:
        prefix = ngram[:-1]
        suffix = ngram[-1]

        if prefix not in transitions:
            transitions[prefix] = []
        transitions[prefix].append(suffix)

    # Generate text
    if not transitions:
        return "No transitions found"

    # Start with random n-gram prefix
    current = random.choice(list(transitions.keys()))
    generated = list(current)

    # Generate words
    for _ in range(length - n + 1):
        if current in transitions:
            next_word = random.choice(transitions[current])
            generated.append(next_word)
            # Update current to be the last n-1 words
            current = tuple(generated[-(n-1):])
        else:
            break

    return " ".join(generated)

# Example text generation
training_corpus = """
Machine learning is a powerful technology that enables computers to learn patterns from data.
Artificial intelligence systems can process natural language and understand human communication.
Deep learning networks use multiple layers to extract features from complex datasets.
Natural language processing helps computers understand and generate human language effectively.
These technologies are revolutionizing industries and creating new opportunities for innovation.
"""

print(f"\nText Generation Examples:")
print("Original corpus sample:", training_corpus[:100] + "...")

# Generate text with different n-gram sizes
for n in [2, 3]:
    generated = generate_text_ngrams(training_corpus, n=n, length=15)
    print(f"\n{n}-gram generated text:")
    print(generated)
```

## 10. Advanced Analysis and Custom Tools
```python
# Text readability analysis
def calculate_readability(text):
    """Calculate various readability metrics"""

    # Basic text statistics
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Filter out punctuation for word count
    words = [word for word in words if word.isalpha()]

    # Count syllables (simple approximation)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        # Every word has at least one syllable
        return max(1, syllables)

    total_syllables = sum(count_syllables(word) for word in words)

    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_syllables_per_word = total_syllables / len(words) if words else 0

    # Flesch Reading Ease Score
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

    # Flesch-Kincaid Grade Level
    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59

    return {
        'sentences': len(sentences),
        'words': len(words),
        'syllables': total_syllables,
        'avg_sentence_length': avg_sentence_length,
        'avg_syllables_per_word': avg_syllables_per_word,
        'flesch_score': flesch_score,
        'fk_grade_level': fk_grade
    }

# Test readability analysis
sample_text = """
Natural language processing is a fascinating field that combines computer science and linguistics.
It enables computers to understand, interpret, and generate human language in a meaningful way.
Modern NLP systems use machine learning algorithms to process large amounts of text data.
"""

readability = calculate_readability(sample_text)
print("Readability Analysis:")
print(f"Sentences: {readability['sentences']}")
print(f"Words: {readability['words']}")
print(f"Average sentence length: {readability['avg_sentence_length']:.2f}")
print(f"Average syllables per word: {readability['avg_syllables_per_word']:.2f}")
print(f"Flesch Reading Ease: {readability['flesch_score']:.2f}")
print(f"Flesch-Kincaid Grade Level: {readability['fk_grade_level']:.2f}")

# Custom text analysis pipeline
class TextAnalyzer:
    """Comprehensive text analysis tool"""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text):
        """Perform comprehensive text analysis"""

        results = {
            'original_text': text,
            'basic_stats': self._basic_stats(text),
            'sentiment': self._analyze_sentiment(text),
            'keywords': self._extract_keywords(text),
            'readability': calculate_readability(text),
            'pos_distribution': self._pos_distribution(text)
        }

        return results

    def _basic_stats(self, text):
        """Calculate basic text statistics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        chars = len(text)

        return {
            'characters': chars,
            'words': len(words),
            'sentences': len(sentences),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0
        }

    def _analyze_sentiment(self, text):
        """Analyze sentiment"""
        scores = self.sia.polarity_scores(text)

        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'scores': scores
        }

    def _extract_keywords(self, text, num_keywords=5):
        """Extract top keywords"""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]

        freq_dist = FreqDist(words)
        return freq_dist.most_common(num_keywords)

    def _pos_distribution(self, text):
        """Analyze POS tag distribution"""
        words = word_tokenize(text)
        pos_tags = pos_tag(words)

        pos_counts = {}
        for word, pos in pos_tags:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        return pos_counts

# Example usage of comprehensive analyzer
analyzer = TextAnalyzer()

sample_analysis_text = """
Artificial intelligence and machine learning are transforming our world in unprecedented ways.
These technologies enable computers to learn from data and make intelligent decisions.
Natural language processing, a subset of AI, allows machines to understand and generate human language.
The applications are endless: from chatbots to language translation, from sentiment analysis to text summarization.
"""

analysis_result = analyzer.analyze(sample_analysis_text)

print("\nComprehensive Text Analysis:")
print("=" * 50)
print(f"Basic Stats: {analysis_result['basic_stats']}")
print(f"Sentiment: {analysis_result['sentiment']['sentiment']} (confidence: {analysis_result['sentiment']['confidence']:.3f})")
print(f"Top Keywords: {analysis_result['keywords']}")
print(f"Readability Score: {analysis_result['readability']['flesch_score']:.1f}")
print(f"Grade Level: {analysis_result['readability']['fk_grade_level']:.1f}")
```

* * * * *

Summary
=======

- **Comprehensive tokenization** support for words, sentences, and custom patterns
- **Text preprocessing** tools including stopword removal, stemming, and lemmatization
- **POS tagging and NER** for linguistic analysis and information extraction
- **Frequency analysis** with n-grams, collocations, and statistical measures
- **Sentiment analysis** using VADER lexicon for opinion mining
- **Text classification** pipeline with scikit-learn integration
- **Similarity metrics** including Jaccard, edit distance, and cosine similarity
- **Corpus analysis** tools for large-scale text processing
- **Readability assessment** with Flesch scores and grade level calculations
- **Extensible framework** for building custom NLP applications and research tools