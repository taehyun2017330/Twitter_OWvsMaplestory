import json
import nltk
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK packages
nltk.download("stopwords")
nltk.download("punkt")

# Stopwords setup
stop_words_en = set(stopwords.words("english")) | {"RT"}
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
        "하는",
        "안",
        "내",
        "늘",
        "들",
        "아",
        "못",
        "기",
        "거",
        "때",
        "것",
        "저",
        "임",
        "또",
        "더",
        "수",
        "말",
        "중",
        "내",
        "나",
        "함",
        "것",
        "게",
        "해",
        "왜",
        "보고",
        "그",
        "뭐",
        "이",
        "고",
        "좀",
        "네",
        "이제",
        "없다",
        "아니다",
        "있다",
        "안되다",
        "같다",
        "대한",
        "그렇다",
        "정말",
        "이번",
        "그냥",
        "때문",
        "진짜",
        "지금",
        "어떻다",
        "그거",
        "오늘",
        "아마",
        "이렇다",
        "조금",
        "다시",
        "여러",
        "한번",
        "어디",
        "누가",
        "무엇",
        "어떤",
        "대해",
        "이런",
        "그런",
        "다른",
        "어떻게",
        "모든",
        "우리",
        "하지",
        "있는",
        "하는",
        "하는",
        "마음",
        "정도",
        "지난",
        "이미",
        "앞으로",
        "부터",
        "사이",
        "역시",
        "대로",
        "이내",
        "가장",
        "더욱",
        "무엇",
        "대해",
        "이후",
        "경우",
        "그대로",
        "다만",
        "만큼",
        "가지",
        "면서",
        "동안",
        "이후",
        "바로",
        "보다",
        "이나",
        "하다",
        "위해",
        "이다",
        "없다",
        "아니다",
        "있다",
        "된다",
        "하고",
        "한테",
        "까지",
        "따라",
        "대로",
        "면서",
        "비해",
        "서는",
        "으로",
        "로서",
        "로써",
        "에게",
        "에서",
        "이고",
        "인데",
        "처럼",
        "하며",
        "하면",
        "항상",
        "해도",
        "해야",
        "혹은",
        "혹시",
    ]
)

# Initialize Okt tokenizer
okt = Okt()


def load_tweets(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tweets = json.load(f)
    return tweets


def preprocess_tweets(tweets, language="english"):
    tokens = []
    stemmer = PorterStemmer()  # Initialize the stemmer

    if language == "english":
        for tweet in tweets:
            words = nltk.word_tokenize(tweet.lower())
            stemmed_words = [
                stemmer.stem(word)
                for word in words
                if word not in stop_words_en and word.isalpha()
            ]
            tokens.extend(stemmed_words)

    elif language == "korean":
        for tweet in tweets:
            words = okt.pos(tweet, norm=True, stem=True)
            tokens.extend(
                [
                    word
                    for word, tag in words
                    if tag in ["Noun", "Adjective"] and word not in stop_words_kr
                ]
            )
    return tokens


def get_word_frequencies(tokens):
    return Counter(tokens)


def generate_wordcloud(
    frequencies, title, font_path="/System/Library/Fonts/Supplemental/AppleGothic.ttf"
):
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        font_path=font_path,
        min_font_size=10,
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# Load tweets from both games
tweets_maple = load_tweets("메이플.json")
tweets_loa = load_tweets("오버워치.json")

# Preprocess and generate word frequencies
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

# Generate frequency distributions
freq_maple_en = get_word_frequencies(tokens_maple_en)
freq_maple_kr = get_word_frequencies(tokens_maple_kr)
freq_loa_en = get_word_frequencies(tokens_loa_en)
freq_loa_kr = get_word_frequencies(tokens_loa_kr)

# Combine English and Korean frequencies
freq_maple_combined = freq_maple_en + freq_maple_kr
freq_loa_combined = freq_loa_en + freq_loa_kr

# Generate and display word clouds for each game
generate_wordcloud(freq_maple_combined, "MapleStory Tweets")
generate_wordcloud(freq_loa_combined, "Overwatch Tweets")

# Optionally, print word frequencies
print(freq_maple_combined.most_common(300))
print(freq_loa_combined.most_common(300))
