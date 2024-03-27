import json
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from konlpy.tag import Okt

nltk.download("stopwords")
nltk.download("punkt")

# Initialize Okt tokenizer
okt = Okt()

# English stopwords
stop_words_en = set(stopwords.words("english"))
stop_words_en.update(["RT"])  # Add 'RT' to the set of English stopwords

# Korean stopwords
stop_words_kr = set(
    [
        "에서",
        "이",
        "의",
        "로",
        "에",
        "과",
        "도",
        "를",
        "으로",
        "한",
        "하다",
        "와",
        "에게",
        "등",
        "으로부터",
        "이다",
        # Add additional Korean stopwords
        "하는",
        "안",
        "내",
        "늘",
        "들",
        "아",
    ]
)


def load_tweets(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tweets = json.load(f)
    return tweets


def preprocess_tweets(tweets, language="english"):
    tokens = []
    if language == "english":
        stemmer = nltk.stem.PorterStemmer()
        for tweet in tweets:
            tokens += nltk.word_tokenize(tweet.lower())
        filtered_tokens = [w for w in tokens if not w in stop_words_en and w.isalpha()]
        return [stemmer.stem(token) for token in filtered_tokens]
    elif language == "korean":
        for tweet in tweets:
            tokens += okt.morphs(tweet)  # Use Okt for morphological analysis
        filtered_tokens = [w for w in tokens if not w in stop_words_kr and w.isalpha()]
        # Filter nouns and adjectives only, as per previous code example
        return [
            word
            for word, tag in okt.pos(" ".join(filtered_tokens))
            if tag in ["Noun", "Adjective"]
        ]


def generate_wordcloud(
    tokens, title, font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf"
):
    counts = Counter(tokens)
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        font_path=font_path,
        min_font_size=10,
    ).generate_from_frequencies(counts)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# Load tweets from both games
tweets_maple = load_tweets("메이플.json")
tweets_loa = load_tweets("로아.json")

# Preprocess and generate word clouds
tokens_maple_en = preprocess_tweets(
    [tweet["text"] for tweet in tweets_maple if tweet["lang"] == "en"], "english"
)
tokens_maple_kr = preprocess_tweets(
    [tweet["text"] for tweet in tweets_maple if tweet["lang"] == "ko"], "korean"
)
tokens_loa_en = preprocess_tweets(
    [tweet["text"] for tweet in tweets_loa if tweet["lang"] == "en"], "english"
)
tokens_loa_kr = preprocess_tweets(
    [tweet["text"] for tweet in tweets_loa if tweet["lang"] == "ko"], "korean"
)

# Combine English and Korean tokens
tokens_maple_combined = tokens_maple_en + tokens_maple_kr
tokens_loa_combined = tokens_loa_en + tokens_loa_kr

# Generate word clouds for each game
generate_wordcloud(tokens_maple_combined, "MapleStory Tweets")
generate_wordcloud(tokens_loa_combined, "Lost Ark Tweets")
