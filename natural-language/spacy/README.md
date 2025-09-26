# spaCy Quick Reference

spaCy is an industrial-strength natural language processing library designed for production use, offering fast and accurate linguistic annotations including tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and more with pre-trained models for multiple languages.

### Installation
```bash
# Install spaCy
pip install spacy

# Download language models
python -m spacy download en_core_web_sm    # Small English model
python -m spacy download en_core_web_md    # Medium English model
python -m spacy download en_core_web_lg    # Large English model

# Additional language models
python -m spacy download es_core_news_sm   # Spanish
python -m spacy download fr_core_news_sm   # French
python -m spacy download de_core_news_sm   # German

# Optional dependencies
pip install matplotlib pandas scikit-learn
```

### Importing spaCy
```python
import spacy
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load pre-trained model
nlp = spacy.load("en_core_web_sm")

# Check model info
print(f"Model: {nlp.meta['name']}")
print(f"Version: {nlp.meta['version']}")
print(f"Language: {nlp.meta['lang']}")
```

* * * * *

## 1. Basic Text Processing and Tokenization
```python
# Process text with spaCy
text = "Apple Inc. is planning to build a new facility in San Francisco, California."
doc = nlp(text)

# Access tokens
print("Tokens:")
for token in doc:
    print(f"Text: {token.text}")
    print(f"Lemma: {token.lemma_}")
    print(f"POS: {token.pos_}")
    print(f"Tag: {token.tag_}")
    print(f"Dependency: {token.dep_}")
    print(f"Shape: {token.shape_}")
    print(f"Is Alpha: {token.is_alpha}")
    print(f"Is Stop: {token.is_stop}")
    print("-" * 20)

# Sentence segmentation
print("\nSentences:")
for i, sent in enumerate(doc.sents):
    print(f"Sentence {i+1}: {sent.text}")

# Advanced tokenization features
def analyze_tokens(text):
    """Comprehensive token analysis"""
    doc = nlp(text)

    token_info = []
    for token in doc:
        info = {
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'dep': token.dep_,
            'head': token.head.text,
            'is_alpha': token.is_alpha,
            'is_stop': token.is_stop,
            'is_punct': token.is_punct,
            'is_space': token.is_space,
            'like_url': token.like_url,
            'like_email': token.like_email,
            'like_num': token.like_num
        }
        token_info.append(info)

    return token_info

# Example usage
sample_text = "Visit https://spacy.io or email support@spacy.io for more info! The price is $50.99."
token_analysis = analyze_tokens(sample_text)

print("\nDetailed Token Analysis:")
for info in token_analysis:
    if info['text'].strip():  # Skip whitespace tokens
        print(f"{info['text']:12} | {info['pos']:8} | {info['tag']:8} | {info['dep']:12} | Head: {info['head']}")

# Custom tokenization rules
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

def create_custom_tokenizer():
    """Create tokenizer with custom rules"""
    nlp = English()

    # Add custom tokenization rules for social media text
    special_cases = {
        "don't": [{"ORTH": "do"}, {"ORTH": "n't"}],
        "won't": [{"ORTH": "wo"}, {"ORTH": "n't"}],
        ":)": [{"ORTH": ":)"}],
        ":(": [{"ORTH": ":("}],
        "@username": [{"ORTH": "@username"}]
    }

    for text, special_case in special_cases.items():
        nlp.tokenizer.add_special_case(text, special_case)

    return nlp

custom_nlp = create_custom_tokenizer()
social_text = "I don't think this will work :( @username"
custom_doc = custom_nlp(social_text)

print(f"\nCustom tokenization: {[token.text for token in custom_doc]}")
```

## 2. Named Entity Recognition (NER)
```python
# Named Entity Recognition
def extract_entities(text, visualize=False):
    """Extract and analyze named entities"""
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_),
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': getattr(ent, 'confidence', None)
        })

    # Visualize entities (requires displacy)
    if visualize:
        from spacy import displacy
        displacy.render(doc, style="ent", jupyter=False)

    return entities

# Example text with various entity types
entity_text = """
Elon Musk, CEO of Tesla Inc. and SpaceX, was born on June 28, 1971, in Pretoria, South Africa.
He moved to the United States and founded companies like PayPal. Tesla's stock price reached $1,000
in 2021. He owns Twitter now, which he bought for $44 billion.
"""

entities = extract_entities(entity_text)

print("Named Entities Found:")
print("-" * 60)
for ent in entities:
    print(f"Text: {ent['text']:20} | Label: {ent['label']:12} | Description: {ent['description']}")

# Entity frequency analysis
entity_labels = [ent['label'] for ent in entities]
entity_counts = Counter(entity_labels)

print(f"\nEntity Type Distribution:")
for label, count in entity_counts.most_common():
    print(f"{label}: {count}")

# Extract specific entity types
def extract_entity_type(text, entity_type):
    """Extract entities of a specific type"""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    return list(set(entities))  # Remove duplicates

# Extract different entity types
people = extract_entity_type(entity_text, "PERSON")
orgs = extract_entity_type(entity_text, "ORG")
money = extract_entity_type(entity_text, "MONEY")
dates = extract_entity_type(entity_text, "DATE")

print(f"\nPeople: {people}")
print(f"Organizations: {orgs}")
print(f"Money: {money}")
print(f"Dates: {dates}")

# Custom NER training (example setup)
def create_training_data():
    """Create sample training data for custom NER"""
    TRAIN_DATA = [
        ("Apple iPhone 12 costs $699", {"entities": [(0, 5, "COMPANY"), (6, 15, "PRODUCT"), (22, 26, "MONEY")]}),
        ("Google Pixel 6 is priced at $599", {"entities": [(0, 6, "COMPANY"), (7, 15, "PRODUCT"), (28, 32, "MONEY")]}),
        ("Samsung Galaxy S21 sells for $799", {"entities": [(0, 7, "COMPANY"), (8, 19, "PRODUCT"), (30, 34, "MONEY")]})
    ]
    return TRAIN_DATA

# Function to add custom entity labels
def add_custom_ner_component(nlp, training_data):
    """Add custom NER component (simplified example)"""
    # This is a simplified example - actual training requires more setup
    print("Custom NER training data prepared:")
    for text, annotations in training_data:
        print(f"Text: {text}")
        print(f"Entities: {annotations['entities']}")
        print()

training_data = create_training_data()
add_custom_ner_component(nlp, training_data)
```

## 3. Part-of-Speech Tagging and Linguistic Features
```python
# Advanced POS tagging and morphological analysis
def analyze_pos_features(text):
    """Detailed POS and morphological analysis"""
    doc = nlp(text)

    analysis = []
    for token in doc:
        features = {
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,           # General POS tag
            'tag': token.tag_,           # Detailed POS tag
            'morph': str(token.morph),   # Morphological features
            'dep': token.dep_,           # Dependency relation
            'head': token.head.text,     # Syntactic head
            'children': [child.text for child in token.children]
        }
        analysis.append(features)

    return analysis

# Example with complex sentence
complex_text = "The researchers have been carefully analyzing the experimental data collected from multiple sources."
pos_analysis = analyze_pos_features(complex_text)

print("Detailed POS Analysis:")
print("-" * 100)
for token_info in pos_analysis:
    if token_info['text'].strip():
        print(f"{token_info['text']:12} | {token_info['pos']:8} | {token_info['tag']:8} | {token_info['dep']:12} | {token_info['morph']}")

# Extract words by POS
def extract_by_pos(text, target_pos=['NOUN', 'VERB', 'ADJ']):
    """Extract words by part of speech"""
    doc = nlp(text)

    pos_words = {pos: [] for pos in target_pos}

    for token in doc:
        if token.pos_ in target_pos and not token.is_stop and not token.is_punct:
            pos_words[token.pos_].append(token.lemma_.lower())

    # Remove duplicates and sort
    for pos in pos_words:
        pos_words[pos] = sorted(list(set(pos_words[pos])))

    return pos_words

# Extract different word types
sample_text = "The innovative researchers quickly developed groundbreaking algorithms using advanced machine learning techniques."
pos_extracted = extract_by_pos(sample_text)

print(f"\nExtracted by POS:")
for pos, words in pos_extracted.items():
    print(f"{pos}: {words}")

# Analyze sentence complexity
def analyze_sentence_complexity(text):
    """Analyze grammatical complexity of sentences"""
    doc = nlp(text)

    complexity_metrics = []

    for sent in doc.sents:
        sent_doc = nlp(sent.text)

        # Count different elements
        tokens = len([token for token in sent_doc if not token.is_space])
        words = len([token for token in sent_doc if token.is_alpha])
        clauses = len([token for token in sent_doc if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
        subordinate_clauses = len([token for token in sent_doc if token.dep_ in ['advcl', 'ccomp']])

        # Calculate complexity score
        complexity_score = (clauses * 2) + subordinate_clauses + (words / 10)

        metrics = {
            'sentence': sent.text,
            'tokens': tokens,
            'words': words,
            'clauses': clauses,
            'subordinate_clauses': subordinate_clauses,
            'complexity_score': complexity_score
        }
        complexity_metrics.append(metrics)

    return complexity_metrics

# Analyze complexity
complex_sentences = [
    "The cat sat on the mat.",
    "Although the weather was terrible, the team decided to continue with their research project.",
    "The researchers, who had been working on this problem for years, finally discovered a solution that would revolutionize the field."
]

for sentence in complex_sentences:
    complexity = analyze_sentence_complexity(sentence)
    for metric in complexity:
        print(f"Sentence: {metric['sentence'][:50]}...")
        print(f"Complexity Score: {metric['complexity_score']:.2f}")
        print(f"Words: {metric['words']}, Clauses: {metric['clauses']}")
        print()
```

## 4. Dependency Parsing and Syntax Analysis
```python
# Dependency parsing and syntax tree analysis
def analyze_dependencies(text, visualize=False):
    """Analyze dependency relationships"""
    doc = nlp(text)

    # Extract dependency information
    dependencies = []
    for token in doc:
        dep_info = {
            'text': token.text,
            'dep': token.dep_,
            'head': token.head.text,
            'head_pos': token.head.pos_,
            'children': [(child.text, child.dep_) for child in token.children]
        }
        dependencies.append(dep_info)

    # Visualize dependency tree
    if visualize:
        from spacy import displacy
        displacy.render(doc, style="dep", jupyter=False)

    return dependencies

# Example sentence for dependency analysis
dep_text = "The experienced teacher carefully explained the complex mathematical concepts to her attentive students."
dependencies = analyze_dependencies(dep_text)

print("Dependency Analysis:")
print("-" * 80)
for dep in dependencies:
    if dep['text'].strip():
        children_str = ", ".join([f"{child[0]}({child[1]})" for child in dep['children']])
        print(f"{dep['text']:12} --{dep['dep']:>10}--> {dep['head']:12} | Children: {children_str}")

# Extract specific dependency relations
def extract_subjects_objects(text):
    """Extract subjects and objects from text"""
    doc = nlp(text)

    subjects = []
    objects = []

    for token in doc:
        # Subjects (nominal subject, passive nominal subject)
        if token.dep_ in ['nsubj', 'nsubjpass']:
            subjects.append({
                'text': token.text,
                'head': token.head.text,
                'head_pos': token.head.pos_
            })

        # Objects (direct object, indirect object, prepositional object)
        elif token.dep_ in ['dobj', 'iobj', 'pobj']:
            objects.append({
                'text': token.text,
                'type': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_
            })

    return subjects, objects

# Extract subjects and objects
subj_obj_text = "The researcher gave the students detailed explanations about machine learning algorithms."
subjects, objects = extract_subjects_objects(subj_obj_text)

print(f"\nSubjects: {subjects}")
print(f"Objects: {objects}")

# Find noun phrases and verb phrases
def extract_phrases(text):
    """Extract noun phrases and analyze verb patterns"""
    doc = nlp(text)

    # Noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Verb phrases (simplified - collect verbs with their modifiers)
    verb_phrases = []
    for token in doc:
        if token.pos_ == 'VERB':
            # Get verb with its auxiliary verbs and modifiers
            phrase_tokens = [token]

            # Add auxiliary verbs
            for child in token.children:
                if child.dep_ in ['aux', 'auxpass']:
                    phrase_tokens.append(child)

            # Add adverbial modifiers
            for child in token.children:
                if child.dep_ in ['advmod']:
                    phrase_tokens.append(child)

            # Sort by position in sentence
            phrase_tokens.sort(key=lambda x: x.i)
            verb_phrase = " ".join([t.text for t in phrase_tokens])
            verb_phrases.append(verb_phrase)

    return noun_phrases, verb_phrases

# Extract phrases
phrase_text = "The brilliant scientists have been successfully developing innovative solutions for complex problems."
noun_phrases, verb_phrases = extract_phrases(phrase_text)

print(f"\nNoun Phrases: {noun_phrases}")
print(f"Verb Phrases: {verb_phrases}")

# Syntax-based text summarization
def syntax_based_summary(text, num_sentences=2):
    """Create summary based on syntactic importance"""
    doc = nlp(text)

    sentence_scores = []

    for sent in doc.sents:
        sent_doc = nlp(sent.text)

        # Score based on syntactic features
        score = 0

        # More points for sentences with subjects and objects
        for token in sent_doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                score += 2
            elif token.dep_ in ['dobj', 'iobj']:
                score += 1.5
            elif token.pos_ in ['NOUN', 'PROPN']:
                score += 1
            elif token.ent_type_:  # Named entities
                score += 1.5

        # Normalize by sentence length
        normalized_score = score / len(sent_doc) if len(sent_doc) > 0 else 0

        sentence_scores.append({
            'sentence': sent.text,
            'score': normalized_score
        })

    # Sort by score and return top sentences
    sentence_scores.sort(key=lambda x: x['score'], reverse=True)
    summary_sentences = [item['sentence'] for item in sentence_scores[:num_sentences]]

    return " ".join(summary_sentences)

# Test syntax-based summarization
long_text = """
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.
It uses algorithms and statistical models to analyze data patterns.
Deep learning is a more advanced form of machine learning that uses neural networks.
These neural networks have multiple layers that can process complex data.
Applications of machine learning include image recognition, natural language processing, and predictive analytics.
Companies use machine learning to improve their products and services.
The field continues to evolve rapidly with new techniques and applications.
"""

summary = syntax_based_summary(long_text, num_sentences=3)
print(f"\nSyntax-based Summary:")
print(summary)
```

## 5. Text Similarity and Semantic Analysis
```python
# Text similarity using word vectors
def calculate_text_similarity(text1, text2):
    """Calculate similarity between texts using spaCy's word vectors"""
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Document similarity (average of word vectors)
    similarity_score = doc1.similarity(doc2)

    return similarity_score

# Test text similarity
texts = [
    "Machine learning is a powerful technology",
    "AI and ML are transforming industries",
    "The weather is nice today",
    "Deep learning uses neural networks"
]

print("Text Similarity Matrix:")
print("-" * 60)
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i <= j:
            similarity = calculate_text_similarity(text1, text2)
            print(f"Text {i+1} vs Text {j+1}: {similarity:.3f}")

# Word similarity and semantic relationships
def analyze_word_relationships(words):
    """Analyze semantic relationships between words"""
    word_docs = [nlp(word) for word in words]

    print(f"\nWord Similarity Analysis:")
    print("-" * 50)

    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i < j:
                similarity = word_docs[i].similarity(word_docs[j])
                print(f"{word1} <-> {word2}: {similarity:.3f}")

# Test word relationships
related_words = ["king", "queen", "man", "woman", "cat", "dog", "computer", "laptop"]
analyze_word_relationships(related_words)

# Find most similar words to a target
def find_similar_words(target_word, candidate_words, top_k=5):
    """Find most similar words from a list of candidates"""
    target_doc = nlp(target_word)

    similarities = []
    for word in candidate_words:
        word_doc = nlp(word)
        similarity = target_doc.similarity(word_doc)
        similarities.append((word, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# Find words similar to "scientist"
candidate_words = ["researcher", "teacher", "doctor", "engineer", "artist", "writer", "programmer", "analyst"]
similar_to_scientist = find_similar_words("scientist", candidate_words, top_k=3)

print(f"\nWords most similar to 'scientist':")
for word, similarity in similar_to_scientist:
    print(f"{word}: {similarity:.3f}")

# Semantic clustering
def semantic_clustering(words, threshold=0.6):
    """Group words into semantic clusters"""
    word_docs = {word: nlp(word) for word in words}
    clusters = []
    used_words = set()

    for word1 in words:
        if word1 in used_words:
            continue

        cluster = [word1]
        used_words.add(word1)

        for word2 in words:
            if word2 not in used_words:
                similarity = word_docs[word1].similarity(word_docs[word2])
                if similarity >= threshold:
                    cluster.append(word2)
                    used_words.add(word2)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters

# Test semantic clustering
vocabulary = ["car", "automobile", "vehicle", "dog", "puppy", "canine", "book", "novel", "literature", "computer", "laptop", "machine"]
clusters = semantic_clustering(vocabulary, threshold=0.5)

print(f"\nSemantic Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")
```

## 6. Information Extraction and Text Analysis
```python
# Extract structured information from text
def extract_structured_info(text):
    """Extract structured information including entities, relationships, and facts"""
    doc = nlp(text)

    extracted_info = {
        'entities': {},
        'relationships': [],
        'facts': [],
        'statistics': {}
    }

    # Group entities by type
    for ent in doc.ents:
        ent_type = ent.label_
        if ent_type not in extracted_info['entities']:
            extracted_info['entities'][ent_type] = []
        extracted_info['entities'][ent_type].append(ent.text)

    # Extract simple relationships (subject-verb-object)
    for token in doc:
        if token.pos_ == 'VERB':
            subjects = [child for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
            objects = [child for child in token.children if child.dep_ in ['dobj', 'pobj']]

            for subj in subjects:
                for obj in objects:
                    relationship = {
                        'subject': subj.text,
                        'verb': token.text,
                        'object': obj.text
                    }
                    extracted_info['relationships'].append(relationship)

    # Extract numerical facts
    for ent in doc.ents:
        if ent.label_ in ['MONEY', 'PERCENT', 'QUANTITY', 'CARDINAL']:
            # Find context around the number
            start_idx = max(0, ent.start - 3)
            end_idx = min(len(doc), ent.end + 3)
            context = doc[start_idx:end_idx].text

            fact = {
                'value': ent.text,
                'type': ent.label_,
                'context': context.strip()
            }
            extracted_info['facts'].append(fact)

    # Basic statistics
    extracted_info['statistics'] = {
        'total_tokens': len(doc),
        'total_sentences': len(list(doc.sents)),
        'total_entities': len(doc.ents),
        'unique_entity_types': len(set([ent.label_ for ent in doc.ents]))
    }

    return extracted_info

# Test information extraction
news_text = """
Apple Inc. reported quarterly revenue of $81.4 billion, a 8.1% increase from last year.
CEO Tim Cook announced the launch of iPhone 14 in September 2022.
The company's market cap reached $2.8 trillion in August.
Apple employs over 154,000 people worldwide and operates 500+ retail stores globally.
"""

extracted = extract_structured_info(news_text)

print("Extracted Structured Information:")
print("=" * 50)
print(f"Entities: {extracted['entities']}")
print(f"Relationships: {extracted['relationships']}")
print(f"Facts: {extracted['facts']}")
print(f"Statistics: {extracted['statistics']}")

# Custom information extraction patterns
def extract_custom_patterns(text):
    """Extract custom patterns using spaCy's matcher"""
    from spacy.matcher import Matcher

    matcher = Matcher(nlp.vocab)
    doc = nlp(text)

    # Define patterns
    patterns = {
        'COMPANY_REVENUE': [
            [{'ENT_TYPE': 'ORG'}, {'LOWER': {'IN': ['reported', 'posted', 'announced']}},
             {'LOWER': {'IN': ['revenue', 'sales', 'earnings']}}, {'LOWER': 'of'}, {'ENT_TYPE': 'MONEY'}]
        ],
        'PERSON_TITLE': [
            [{'ENT_TYPE': 'PERSON'}, {'POS': 'PUNCT', 'OP': '?'},
             {'LOWER': {'IN': ['ceo', 'president', 'director', 'manager']}}]
        ],
        'DATE_EVENT': [
            [{'LOWER': {'IN': ['launched', 'released', 'announced']}},
             {'POS': {'IN': ['NOUN', 'PROPN']}, 'OP': '+'}, {'LOWER': 'in'}, {'ENT_TYPE': 'DATE'}]
        ]
    }

    # Add patterns to matcher
    for pattern_name, pattern_list in patterns.items():
        matcher.add(pattern_name, pattern_list)

    # Find matches
    matches = matcher(doc)

    extracted_patterns = []
    for match_id, start, end in matches:
        pattern_name = nlp.vocab.strings[match_id]
        matched_text = doc[start:end].text
        extracted_patterns.append({
            'pattern': pattern_name,
            'text': matched_text,
            'start': start,
            'end': end
        })

    return extracted_patterns

# Test custom pattern extraction
custom_patterns = extract_custom_patterns(news_text)
print(f"\nCustom Pattern Matches:")
for pattern in custom_patterns:
    print(f"Pattern: {pattern['pattern']}")
    print(f"Text: {pattern['text']}")
    print()

# Timeline extraction
def extract_timeline(text):
    """Extract temporal information and create timeline"""
    doc = nlp(text)

    timeline_events = []

    for sent in doc.sents:
        sent_doc = nlp(sent.text)

        # Find dates and events in the sentence
        dates = [ent for ent in sent_doc.ents if ent.label_ in ['DATE', 'TIME']]
        events = []

        # Extract main verbs as events
        for token in sent_doc:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                # Get subject and object for context
                subjects = [child.text for child in token.children if child.dep_ in ['nsubj', 'nsubjpass']]
                objects = [child.text for child in token.children if child.dep_ in ['dobj', 'pobj']]

                event_description = f"{' '.join(subjects)} {token.text} {' '.join(objects)}".strip()
                events.append(event_description)

        # Combine dates with events
        for date in dates:
            for event in events:
                timeline_events.append({
                    'date': date.text,
                    'event': event,
                    'sentence': sent.text.strip()
                })

    return timeline_events

# Extract timeline
timeline = extract_timeline(news_text)
print(f"Timeline Events:")
for event in timeline:
    print(f"Date: {event['date']} | Event: {event['event']}")
```

## 7. Text Classification and Document Analysis
```python
# Document classification using spaCy features
def extract_document_features(text):
    """Extract features for document classification"""
    doc = nlp(text)

    features = {
        # Basic statistics
        'doc_length': len(doc),
        'sentence_count': len(list(doc.sents)),
        'avg_sentence_length': len(doc) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,

        # POS distribution
        'noun_ratio': len([token for token in doc if token.pos_ == 'NOUN']) / len(doc),
        'verb_ratio': len([token for token in doc if token.pos_ == 'VERB']) / len(doc),
        'adj_ratio': len([token for token in doc if token.pos_ == 'ADJ']) / len(doc),

        # Entity features
        'entity_count': len(doc.ents),
        'person_count': len([ent for ent in doc.ents if ent.label_ == 'PERSON']),
        'org_count': len([ent for ent in doc.ents if ent.label_ == 'ORG']),
        'money_count': len([ent for ent in doc.ents if ent.label_ == 'MONEY']),

        # Lexical features
        'unique_words': len(set([token.lemma_.lower() for token in doc if token.is_alpha])),
        'stopword_ratio': len([token for token in doc if token.is_stop]) / len(doc),

        # Syntactic complexity
        'avg_tree_depth': np.mean([len(list(token.ancestors)) for token in doc]),
        'dependency_variety': len(set([token.dep_ for token in doc])),
    }

    return features

# Example document classification
def classify_document_type(text):
    """Simple document type classification based on features"""
    features = extract_document_features(text)

    # Simple rule-based classification
    if features['money_count'] > 0 and features['org_count'] > 0:
        doc_type = 'Business/Financial'
    elif features['person_count'] > features['org_count'] and features['entity_count'] > 3:
        doc_type = 'Biography/News'
    elif features['adj_ratio'] > 0.15 and features['entity_count'] < 2:
        doc_type = 'Descriptive/Literary'
    elif features['noun_ratio'] > 0.3 and features['dependency_variety'] > 15:
        doc_type = 'Technical/Academic'
    else:
        doc_type = 'General'

    return doc_type, features

# Test document classification
sample_documents = [
    "Apple Inc. reported record quarterly revenue of $123.9 billion, exceeding analyst expectations.",
    "The beautiful sunset painted the sky in brilliant shades of orange and pink as the day came to a peaceful end.",
    "Machine learning algorithms utilize statistical models to identify patterns in large datasets and make predictions.",
    "Barack Obama served as the 44th President of the United States from 2009 to 2017."
]

print("Document Classification:")
print("-" * 60)
for i, doc in enumerate(sample_documents):
    doc_type, features = classify_document_type(doc)
    print(f"Document {i+1}: {doc_type}")
    print(f"Sample: {doc[:50]}...")
    print(f"Key features: Entities={features['entity_count']}, Noun ratio={features['noun_ratio']:.2f}")
    print()

# Topic modeling preparation
def prepare_for_topic_modeling(texts):
    """Prepare texts for topic modeling by extracting meaningful tokens"""
    processed_docs = []

    for text in texts:
        doc = nlp(text)

        # Extract meaningful tokens
        tokens = []
        for token in doc:
            # Include nouns, adjectives, and verbs
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and
                not token.is_stop and
                not token.is_punct and
                not token.is_space and
                len(token.text) > 2):
                tokens.append(token.lemma_.lower())

        processed_docs.append(tokens)

    return processed_docs

# Test topic modeling preparation
topic_texts = [
    "Machine learning algorithms can automatically learn patterns from data without explicit programming.",
    "Deep neural networks have revolutionized computer vision and image recognition tasks.",
    "Natural language processing enables computers to understand and generate human language.",
    "Artificial intelligence systems are being deployed in healthcare, finance, and transportation."
]

topic_docs = prepare_for_topic_modeling(topic_texts)
print("Prepared for Topic Modeling:")
for i, doc in enumerate(topic_docs):
    print(f"Doc {i+1}: {doc}")
```

## 8. Custom Pipeline Components and Extensions
```python
# Create custom pipeline component
from spacy.language import Language
from spacy.tokens import Doc, Token

# Custom component for text statistics
@Language.component("text_stats")
def text_statistics_component(doc):
    """Add text statistics to doc"""

    # Calculate statistics
    word_count = len([token for token in doc if token.is_alpha])
    sentence_count = len(list(doc.sents))
    avg_word_length = np.mean([len(token.text) for token in doc if token.is_alpha])
    complexity_score = len(set([token.dep_ for token in doc])) / len(doc) if len(doc) > 0 else 0

    # Add custom attributes
    Doc.set_extension("word_count", default=0, force=True)
    Doc.set_extension("sentence_count", default=0, force=True)
    Doc.set_extension("avg_word_length", default=0.0, force=True)
    Doc.set_extension("complexity_score", default=0.0, force=True)

    doc._.word_count = word_count
    doc._.sentence_count = sentence_count
    doc._.avg_word_length = avg_word_length
    doc._.complexity_score = complexity_score

    return doc

# Custom component for sentiment scoring
@Language.component("simple_sentiment")
def simple_sentiment_component(doc):
    """Add simple sentiment analysis to doc"""

    # Simple positive/negative word lists
    positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best'])
    negative_words = set(['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing'])

    positive_count = 0
    negative_count = 0

    for token in doc:
        if token.lemma_.lower() in positive_words:
            positive_count += 1
        elif token.lemma_.lower() in negative_words:
            negative_count += 1

    # Calculate sentiment score
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
    else:
        sentiment_score = 0

    # Add custom attributes
    Doc.set_extension("sentiment_score", default=0.0, force=True)
    Doc.set_extension("positive_words", default=0, force=True)
    Doc.set_extension("negative_words", default=0, force=True)

    doc._.sentiment_score = sentiment_score
    doc._.positive_words = positive_count
    doc._.negative_words = negative_count

    return doc

# Create custom pipeline
def create_custom_pipeline():
    """Create spaCy pipeline with custom components"""
    # Start with blank model or load existing
    custom_nlp = spacy.load("en_core_web_sm")

    # Add custom components
    custom_nlp.add_pipe("text_stats", last=True)
    custom_nlp.add_pipe("simple_sentiment", last=True)

    return custom_nlp

# Test custom pipeline
custom_nlp = create_custom_pipeline()

test_texts = [
    "This is an excellent movie with amazing acting and wonderful storytelling!",
    "The weather is terrible today and I hate this awful rain.",
    "Machine learning algorithms can process large datasets efficiently."
]

print("Custom Pipeline Results:")
print("-" * 50)
for text in test_texts:
    doc = custom_nlp(text)

    print(f"Text: {text}")
    print(f"Word count: {doc._.word_count}")
    print(f"Sentence count: {doc._.sentence_count}")
    print(f"Avg word length: {doc._.avg_word_length:.2f}")
    print(f"Complexity score: {doc._.complexity_score:.3f}")
    print(f"Sentiment score: {doc._.sentiment_score:.2f}")
    print(f"Positive words: {doc._.positive_words}, Negative words: {doc._.negative_words}")
    print()

# Custom matcher for complex patterns
def create_advanced_matcher():
    """Create matcher with advanced patterns"""
    from spacy.matcher import Matcher, DependencyMatcher

    matcher = Matcher(nlp.vocab)
    dep_matcher = DependencyMatcher(nlp.vocab)

    # Pattern for finding comparative statements
    comparative_pattern = [
        [{"POS": "NOUN"}, {"LOWER": {"IN": ["is", "are"]}},
         {"POS": "ADJ", "DEP": {"IN": ["acomp", "attr"]}},
         {"LOWER": "than"}, {"POS": "NOUN"}]
    ]
    matcher.add("COMPARATIVE", comparative_pattern)

    # Dependency pattern for subject-verb-object relationships
    svo_pattern = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "subject",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "object",
            "RIGHT_ATTRS": {"DEP": "dobj"}
        }
    ]
    dep_matcher.add("SVO", [svo_pattern])

    return matcher, dep_matcher

# Test advanced matching
matcher, dep_matcher = create_advanced_matcher()

complex_text = "Machine learning is more powerful than traditional programming. The algorithm processes the data efficiently."
doc = nlp(complex_text)

# Find token-based matches
matches = matcher(doc)
print("Token-based Matches:")
for match_id, start, end in matches:
    pattern_name = nlp.vocab.strings[match_id]
    matched_span = doc[start:end]
    print(f"Pattern: {pattern_name}, Match: {matched_span.text}")

# Find dependency-based matches
dep_matches = dep_matcher(doc)
print("\nDependency-based Matches:")
for match_id, token_ids in dep_matches:
    pattern_name = nlp.vocab.strings[match_id]
    tokens = [doc[i] for i in token_ids]
    print(f"Pattern: {pattern_name}, Tokens: {[token.text for token in tokens]}")
```

## 9. Multi-language Support and Cross-lingual Analysis
```python
# Multi-language processing
def load_multilingual_models():
    """Load multiple language models"""
    models = {}

    try:
        models['en'] = spacy.load("en_core_web_sm")
        print("English model loaded")
    except OSError:
        print("English model not found")

    try:
        models['es'] = spacy.load("es_core_news_sm")
        print("Spanish model loaded")
    except OSError:
        print("Spanish model not found - install with: python -m spacy download es_core_news_sm")

    try:
        models['fr'] = spacy.load("fr_core_news_sm")
        print("French model loaded")
    except OSError:
        print("French model not found - install with: python -m spacy download fr_core_news_sm")

    return models

# Cross-lingual text analysis
def analyze_multilingual_text(texts_with_langs, models):
    """Analyze texts in multiple languages"""
    results = {}

    for lang, text in texts_with_langs.items():
        if lang in models:
            nlp_model = models[lang]
            doc = nlp_model(text)

            analysis = {
                'language': lang,
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'pos_distribution': Counter([token.pos_ for token in doc]),
                'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
                'sentence_count': len(list(doc.sents))
            }
            results[lang] = analysis
        else:
            results[lang] = {'error': f'Model for {lang} not available'}

    return results

# Test multilingual analysis
multilingual_models = load_multilingual_models()

multilingual_texts = {
    'en': "Apple Inc. is a technology company founded by Steve Jobs in California.",
    'es': "Apple Inc. es una empresa de tecnología fundada por Steve Jobs en California.",
    'fr': "Apple Inc. est une entreprise de technologie fondée par Steve Jobs en Californie."
}

if multilingual_models:
    multilingual_results = analyze_multilingual_text(multilingual_texts, multilingual_models)

    print("Multilingual Analysis Results:")
    print("-" * 60)
    for lang, analysis in multilingual_results.items():
        if 'error' not in analysis:
            print(f"Language: {lang.upper()}")
            print(f"Entities: {analysis['entities']}")
            print(f"Noun Phrases: {analysis['noun_phrases']}")
            print(f"Sentences: {analysis['sentence_count']}")
            print()

# Language detection using spaCy
def detect_language_spacy(text, models):
    """Simple language detection by trying different models"""
    scores = {}

    for lang, model in models.items():
        try:
            doc = model(text)
            # Simple scoring based on successful entity recognition
            entity_score = len(doc.ents)
            # Score based on POS tag variety (more variety = better fit)
            pos_variety = len(set([token.pos_ for token in doc]))

            total_score = entity_score + (pos_variety / 10)
            scores[lang] = total_score
        except:
            scores[lang] = 0

    if scores:
        detected_lang = max(scores, key=scores.get)
        confidence = scores[detected_lang] / sum(scores.values()) if sum(scores.values()) > 0 else 0
        return detected_lang, confidence, scores
    else:
        return None, 0, {}

# Test language detection
test_sentences = [
    "Hello, how are you today?",
    "Hola, ¿cómo estás hoy?",
    "Bonjour, comment allez-vous aujourd'hui?"
]

if multilingual_models:
    print("Language Detection:")
    print("-" * 40)
    for sentence in test_sentences:
        detected_lang, confidence, scores = detect_language_spacy(sentence, multilingual_models)
        print(f"Text: {sentence}")
        print(f"Detected: {detected_lang} (confidence: {confidence:.2f})")
        print(f"Scores: {scores}")
        print()

# Cross-lingual entity alignment
def align_entities_across_languages(texts_with_langs, models):
    """Find equivalent entities across different languages"""
    all_entities = {}

    # Extract entities from each language
    for lang, text in texts_with_langs.items():
        if lang in models:
            doc = models[lang](text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            all_entities[lang] = entities

    # Simple entity alignment (by type and position)
    aligned_entities = {}
    if 'en' in all_entities:  # Use English as reference
        for i, (en_text, en_label) in enumerate(all_entities['en']):
            aligned_entities[i] = {'en': (en_text, en_label)}

            # Find corresponding entities in other languages
            for lang, entities in all_entities.items():
                if lang != 'en' and i < len(entities):
                    aligned_entities[i][lang] = entities[i]

    return aligned_entities

# Test cross-lingual entity alignment
if multilingual_models and len(multilingual_models) > 1:
    aligned = align_entities_across_languages(multilingual_texts, multilingual_models)

    print("Cross-lingual Entity Alignment:")
    print("-" * 50)
    for idx, entities in aligned.items():
        print(f"Entity group {idx + 1}:")
        for lang, (text, label) in entities.items():
            print(f"  {lang.upper()}: {text} ({label})")
        print()
```

## 10. Performance Optimization and Production Deployment
```python
# Performance optimization techniques
import time
from functools import lru_cache

# Efficient processing for large texts
def optimize_large_text_processing():
    """Demonstrate optimization techniques for large texts"""

    # Use nlp.pipe() for batch processing
    def batch_process_texts(texts, batch_size=50):
        """Process multiple texts efficiently"""
        start_time = time.time()

        # Process in batches
        results = []
        for doc in nlp.pipe(texts, batch_size=batch_size):
            # Extract key information
            info = {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
            }
            results.append(info)

        end_time = time.time()
        print(f"Batch processed {len(texts)} texts in {end_time - start_time:.2f} seconds")

        return results

    # Compare with individual processing
    def individual_process_texts(texts):
        """Process texts individually (slower)"""
        start_time = time.time()

        results = []
        for text in texts:
            doc = nlp(text)
            info = {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
            }
            results.append(info)

        end_time = time.time()
        print(f"Individually processed {len(texts)} texts in {end_time - start_time:.2f} seconds")

        return results

    # Test with sample texts
    sample_texts = [
        "Apple Inc. reported strong quarterly earnings.",
        "Google announced new AI research initiatives.",
        "Microsoft is investing heavily in cloud computing.",
        "Amazon continues to expand its e-commerce platform.",
        "Tesla is revolutionizing the automotive industry."
    ] * 20  # 100 texts total

    print("Performance Comparison:")
    batch_results = batch_process_texts(sample_texts)
    individual_results = individual_process_texts(sample_texts)

    return batch_results

# Memory-efficient processing
def memory_efficient_processing():
    """Demonstrate memory-efficient text processing"""

    # Disable unnecessary pipeline components
    def create_lightweight_nlp():
        """Create lightweight nlp pipeline"""
        # Load only required components
        nlp_light = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return nlp_light

    # Process with limited components
    nlp_light = create_lightweight_nlp()

    test_text = "This is a sample text for testing lightweight processing."

    # Compare processing times
    start_time = time.time()
    doc_full = nlp(test_text)
    full_time = time.time() - start_time

    start_time = time.time()
    doc_light = nlp_light(test_text)
    light_time = time.time() - start_time

    print(f"Full pipeline: {full_time:.4f} seconds")
    print(f"Lightweight pipeline: {light_time:.4f} seconds")
    print(f"Speedup: {full_time / light_time:.2f}x faster")

    return nlp_light

# Caching for repeated processing
@lru_cache(maxsize=1000)
def cached_entity_extraction(text):
    """Cached entity extraction for repeated texts"""
    doc = nlp(text)
    return tuple([(ent.text, ent.label_) for ent in doc.ents])

def test_caching():
    """Test caching performance"""
    repeated_texts = [
        "Apple Inc. is a technology company.",
        "Google develops search technologies.",
        "Microsoft creates software products."
    ] * 100  # Many repetitions

    # Without caching
    start_time = time.time()
    for text in repeated_texts:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    no_cache_time = time.time() - start_time

    # With caching
    start_time = time.time()
    for text in repeated_texts:
        entities = cached_entity_extraction(text)
    cache_time = time.time() - start_time

    print(f"Without caching: {no_cache_time:.4f} seconds")
    print(f"With caching: {cache_time:.4f} seconds")
    print(f"Speedup: {no_cache_time / cache_time:.2f}x faster")

# Production deployment utilities
def create_production_nlp_service():
    """Create production-ready NLP service"""

    class NLPService:
        def __init__(self, model_name="en_core_web_sm", batch_size=32):
            self.nlp = spacy.load(model_name)
            self.batch_size = batch_size

            # Add custom attributes for service info
            self.model_info = {
                'name': model_name,
                'version': self.nlp.meta['version'],
                'batch_size': batch_size
            }

        def analyze_text(self, text):
            """Analyze single text"""
            doc = self.nlp(text)

            return {
                'entities': [{'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char}
                           for ent in doc.ents],
                'tokens': [{'text': token.text, 'pos': token.pos_, 'lemma': token.lemma_} for token in doc],
                'sentences': [sent.text for sent in doc.sents]
            }

        def analyze_batch(self, texts):
            """Analyze multiple texts efficiently"""
            results = []

            for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
                result = {
                    'entities': [{'text': ent.text, 'label': ent.label_} for ent in doc.ents],
                    'sentence_count': len(list(doc.sents)),
                    'token_count': len(doc)
                }
                results.append(result)

            return results

        def get_service_info(self):
            """Get service information"""
            return self.model_info

        def health_check(self):
            """Health check for service monitoring"""
            try:
                test_doc = self.nlp("Test sentence.")
                return {'status': 'healthy', 'model_loaded': True}
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}

    return NLPService()

# Test production service
def test_production_service():
    """Test production NLP service"""
    service = create_production_nlp_service()

    # Health check
    health = service.health_check()
    print(f"Service Health: {health}")

    # Service info
    info = service.get_service_info()
    print(f"Service Info: {info}")

    # Single text analysis
    result = service.analyze_text("Apple Inc. announced new products today.")
    print(f"Single Analysis: {result}")

    # Batch analysis
    batch_texts = [
        "Google releases new AI model.",
        "Microsoft updates Windows operating system.",
        "Amazon expands cloud services."
    ]
    batch_results = service.analyze_batch(batch_texts)
    print(f"Batch Analysis: {len(batch_results)} results generated")

    return service

# Run optimization tests
print("=== Performance Optimization Tests ===")
optimize_large_text_processing()

print("\n=== Memory Efficiency Tests ===")
memory_efficient_processing()

print("\n=== Caching Tests ===")
test_caching()

print("\n=== Production Service Tests ===")
production_service = test_production_service()

# Model serialization for deployment
def save_custom_model(nlp_model, output_path):
    """Save custom model for deployment"""
    nlp_model.to_disk(output_path)
    print(f"Model saved to {output_path}")

def load_custom_model(model_path):
    """Load custom model from disk"""
    nlp_loaded = spacy.load(model_path)
    print(f"Model loaded from {model_path}")
    return nlp_loaded

# Example model operations
print("\n=== Model Serialization ===")
# save_custom_model(nlp, "./custom_spacy_model")
# loaded_nlp = load_custom_model("./custom_spacy_model")
print("Model serialization examples ready (commented out to avoid file operations)")
```

* * * * *

Summary
=======

- **Industrial-strength NLP** with pre-trained models for 60+ languages
- **Complete linguistic pipeline** including tokenization, POS tagging, NER, and dependency parsing
- **Advanced entity recognition** with customizable patterns and training capabilities
- **Semantic analysis** using word vectors and document similarity
- **Production-ready features** including batch processing and memory optimization
- **Extensible architecture** with custom pipeline components and matcher patterns
- **Multi-language support** for cross-lingual analysis and entity alignment
- **Performance optimization** tools for large-scale text processing
- **Easy deployment** with serialization and service patterns
- **Rich linguistic features** including morphology, syntax trees, and noun phrases