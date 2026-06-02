import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Optional heavy dependencies (graceful fallback) ─────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.probability import FreqDist
    from nltk.util import ngrams
    # Download silently
    for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                "punkt_tab", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    NLTK_OK = True
except ImportError:
    NLTK_OK = False
    print("NLTK not installed. Install: pip install nltk")

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("scikit-learn not installed. Install: pip install scikit-learn")

try:
    from wordcloud import WordCloud
    WC_OK = True
except ImportError:
    WC_OK = False
    print("wordcloud not installed. Install: pip install wordcloud")

plt.rcParams["figure.dpi"] = 100
sns.set_theme(style="whitegrid")

# ════════════════════════════════════════════════════════════
# CORPUS
# ════════════════════════════════════════════════════════════
corpus = [
    "Natural language processing (NLP) is a subfield of linguistics and artificial intelligence.",
    "Machine learning algorithms can process and analyze large amounts of text data efficiently.",
    "Deep learning models like BERT and GPT have revolutionized natural language understanding.",
    "Text preprocessing involves tokenization, stemming, lemmatization, and stop-word removal.",
    "Sentiment analysis determines whether a piece of text expresses positive, negative, or neutral opinions.",
    "Named entity recognition identifies entities such as persons, organizations, and locations in text.",
    "Text classification assigns predefined categories to documents based on their content.",
    "Word embeddings like Word2Vec and GloVe represent words as dense numerical vectors.",
    "Topic modeling discovers hidden thematic structure in large document collections.",
    "Information retrieval systems index and search documents using TF-IDF and BM25 algorithms.",
]

reviews = [
    "This product is absolutely amazing! I love it so much.",
    "Terrible quality. Completely waste of money. Very disappointed.",
    "It's okay, not great but not bad either. Average experience.",
    "Fantastic! Best purchase I have made this year. Highly recommended.",
    "Poor customer service and the product broke after one day.",
    "Decent product for the price. Would consider buying again.",
    "Outstanding performance! Exceeded all my expectations.",
    "Not worth the price at all. Better alternatives available.",
]

# ── 1. Basic String Operations ──────────────────────────────────────────────
print("=" * 60)
print("1. Basic String Operations")
print("=" * 60)
sample = corpus[0]
print("Original  :", sample)
print("Lower     :", sample.lower())
print("Upper     :", sample.upper())
print("Length    :", len(sample))
print("Word count:", len(sample.split()))
print("Replace   :", sample.replace("NLP", "Natural Language Processing"))
print("Startswith:", sample.startswith("Natural"))
print("Split     :", sample.split()[:6], "...")

# ── 2. Regular Expressions ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Regular Expressions")
print("=" * 60)
text = ("Contact us: john.doe@example.com or support@nlp.org. "
        "Visit https://www.nlptools.com or http://example.org. "
        "Call +1-800-555-1234 or (022) 9876-5432.")
emails  = re.findall(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
urls    = re.findall(r"https?://[^\s]+", text)
phones  = re.findall(r"[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]", text)
print("Emails  :", emails)
print("URLs    :", urls)
print("Phones  :", phones)

clean = re.sub(r"[^a-zA-Z\s]", "", text)
print("Cleaned :", clean[:80], "...")

# ── 3. Text Preprocessing (from scratch) ───────────────────────────────────
print("\n" + "=" * 60)
print("3. Text Preprocessing (manual)")
print("=" * 60)
STOPWORDS_SIMPLE = {"is","a","and","of","the","in","to","for","on","at","by",
                    "an","are","as","be","it","or","this","that","these","those",
                    "was","were","with","have","has","had","will","can","may",
                    "not","from","but","its"}

def preprocess(text, remove_stops=True):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    if remove_stops:
        tokens = [t for t in tokens if t not in STOPWORDS_SIMPLE]
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

print("Sample text:", corpus[0])
tokens = preprocess(corpus[0])
print("Tokens:", tokens)

all_tokens = []
for doc in corpus:
    all_tokens.extend(preprocess(doc))
print("\nTotal tokens across corpus:", len(all_tokens))
print("Unique tokens            :", len(set(all_tokens)))

# ── 4. Word Frequency Analysis ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Word Frequency Analysis")
print("=" * 60)
freq = Counter(all_tokens)
top20 = freq.most_common(20)
print("Top 20 words:")
for word, count in top20:
    print(f"  {word:25s}: {count}")

words, counts = zip(*top20)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(words[::-1], counts[::-1], color=sns.color_palette("Blues_d", 20))
ax.set_title("Top 20 Most Frequent Words"); ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p10_1_word_freq.png", dpi=100); plt.show()
print("Word frequency chart saved.")

# ── 5. N-grams ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. N-gram Analysis")
print("=" * 60)
bigrams  = [(all_tokens[i], all_tokens[i+1])
            for i in range(len(all_tokens)-1)]
trigrams = [(all_tokens[i], all_tokens[i+1], all_tokens[i+2])
            for i in range(len(all_tokens)-2)]

print("Top 10 Bigrams:")
for bg, cnt in Counter(bigrams).most_common(10):
    print(f"  {' '.join(bg):30s}: {cnt}")
print("\nTop 10 Trigrams:")
for tg, cnt in Counter(trigrams).most_common(10):
    print(f"  {' '.join(tg):40s}: {cnt}")

# ── 6. NLTK – Advanced Preprocessing ───────────────────────────────────────
if NLTK_OK:
    print("\n" + "=" * 60)
    print("6. NLTK – Tokenization, Stemming & Lemmatization")
    print("=" * 60)
    sample = corpus[3]
    word_tokens = word_tokenize(sample)
    sent_tokens = sent_tokenize(" ".join(corpus))

    stop_words = set(stopwords.words("english"))
    filtered   = [w for w in word_tokens
                  if w.lower() not in stop_words and w not in string.punctuation]

    stemmer    = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed    = [stemmer.stem(w)    for w in filtered]
    lemmatized = [lemmatizer.lemmatize(w.lower()) for w in filtered]

    print("Original    :", word_tokens)
    print("Filtered    :", filtered)
    print("Stemmed     :", stemmed)
    print("Lemmatized  :", lemmatized)
    print("Sentence count:", len(sent_tokens))

    # POS Tagging
    pos_tags = nltk.pos_tag(filtered)
    print("\nPOS Tags:", pos_tags)

    # Frequency Distribution Plot
    all_nltk_tokens = []
    for doc in corpus:
        toks = word_tokenize(doc.lower())
        all_nltk_tokens += [t for t in toks
                            if t.isalpha() and t not in stop_words]
    fdist = FreqDist(all_nltk_tokens)
    fig, ax = plt.subplots(figsize=(10, 4))
    fdist.plot(20, cumulative=False, show=False)
    ax = plt.gca()
    ax.set_title("NLTK Frequency Distribution – Top 20")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/p10_2_nltk_freqdist.png", dpi=100); plt.show()
    print("NLTK frequency distribution saved.")

# ── 7. TF-IDF ───────────────────────────────────────────────────────────────
if SKLEARN_OK:
    print("\n" + "=" * 60)
    print("7. TF-IDF Representation")
    print("=" * 60)
    tfidf = TfidfVectorizer(max_features=20, stop_words="english")
    X_tfidf = tfidf.fit_transform(corpus)
    tfidf_df = pd.DataFrame(X_tfidf.toarray(),
                             columns=tfidf.get_feature_names_out(),
                             index=[f"Doc{i+1}" for i in range(len(corpus))])
    print("TF-IDF Matrix (shape):", X_tfidf.shape)
    print(tfidf_df.round(3))

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(tfidf_df, cmap="YlOrRd", linewidths=0.3, annot=True,
                fmt=".2f", ax=ax, cbar_kws={"label":"TF-IDF Score"})
    ax.set_title("TF-IDF Score per Document (Top 20 Terms)")
    ax.set_xlabel("Terms"); ax.set_ylabel("Documents")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/p10_3_tfidf.png", dpi=100); plt.show()
    print("TF-IDF heatmap saved.")

    # ── 8. Cosine Similarity ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("8. Document Similarity (Cosine)")
    print("=" * 60)
    cos_sim = cosine_similarity(X_tfidf)
    cos_df  = pd.DataFrame(cos_sim,
                            index=[f"D{i+1}" for i in range(len(corpus))],
                            columns=[f"D{j+1}" for j in range(len(corpus))])
    print(cos_df.round(3))

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(corpus), dtype=bool)
    sns.heatmap(cos_df, annot=True, fmt=".2f", cmap="Blues",
                mask=mask, linewidths=0.3, ax=ax)
    ax.set_title("Document Cosine Similarity Matrix")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/p10_4_cosine_sim.png", dpi=100); plt.show()
    print("Cosine similarity heatmap saved.")

# ── 9. Simple Sentiment Analysis (Lexicon-based) ────────────────────────────
print("\n" + "=" * 60)
print("9. Lexicon-based Sentiment Analysis")
print("=" * 60)
POSITIVE_WORDS = {"amazing","love","fantastic","outstanding","great","excellent",
                  "best","recommended","highly","good","decent","wonderful","nice"}
NEGATIVE_WORDS = {"terrible","waste","disappointed","poor","broke","bad","worst",
                  "awful","horrible","cheap","broken","useless","toxic"}

def simple_sentiment(text):
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos > neg:
        return "Positive", pos, neg
    elif neg > pos:
        return "Negative", pos, neg
    else:
        return "Neutral", pos, neg

print(f"{'Review':<55} {'Sentiment':<10} {'Pos':>3} {'Neg':>3}")
print("-" * 75)
sentiments = []
for rev in reviews:
    label, pos, neg = simple_sentiment(rev)
    sentiments.append(label)
    print(f"{rev[:53]:<55} {label:<10} {pos:>3} {neg:>3}")

# Sentiment distribution
sent_counts = Counter(sentiments)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(sent_counts.keys(), sent_counts.values(),
       color=["steelblue","tomato","gray"])
ax.set_title("Sentiment Distribution"); ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p10_5_sentiment.png", dpi=100); plt.show()
print("Sentiment chart saved.")

# ── 10. Word Cloud ───────────────────────────────────────────────────────────
if WC_OK:
    print("\n" + "=" * 60)
    print("10. Word Cloud")
    print("=" * 60)
    full_text = " ".join(corpus + reviews)
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap="viridis", max_words=80,
                   stopwords=set(STOPWORDS_SIMPLE)).generate(full_text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off"); ax.set_title("Word Cloud – Corpus + Reviews", fontsize=14)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/p10_6_wordcloud.png", dpi=100); plt.show()
    print("Word cloud saved.")
else:
    print("ℹ️  Skipping word cloud (wordcloud not installed).")

# ── 11. Text Statistics Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Corpus-Level Text Statistics")
print("=" * 60)
records = []
for i, doc in enumerate(corpus):
    words_raw = doc.split()
    sentences = re.split(r"[.!?]+", doc)
    sentences = [s.strip() for s in sentences if s.strip()]
    chars_no_space = len(doc.replace(" ", ""))
    records.append({
        "Doc": f"Doc{i+1}",
        "Char_Count":  len(doc),
        "Word_Count":  len(words_raw),
        "Sent_Count":  len(sentences),
        "Avg_Word_Len": round(chars_no_space / len(words_raw), 2),
        "Unique_Words": len(set(w.lower().strip(string.punctuation)
                                for w in words_raw)),
        "Lexical_Diversity": round(len(set(words_raw)) / len(words_raw), 3),
    })
stats_df = pd.DataFrame(records)
print(stats_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(stats_df["Doc"], stats_df["Word_Count"],
            color=sns.color_palette("Blues_d", len(corpus)))
axes[0].set_title("Word Count per Doc")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(stats_df["Doc"], stats_df["Unique_Words"],
            color=sns.color_palette("Greens_d", len(corpus)))
axes[1].set_title("Unique Words per Doc")
axes[1].tick_params(axis="x", rotation=45)

axes[2].bar(stats_df["Doc"], stats_df["Lexical_Diversity"],
            color=sns.color_palette("Oranges_d", len(corpus)))
axes[2].set_title("Lexical Diversity per Doc")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/p10_7_text_stats.png", dpi=100); plt.show()
print("Text statistics chart saved.")

print("\n✅ Practical 10 Complete! All text analysis outputs saved.")
