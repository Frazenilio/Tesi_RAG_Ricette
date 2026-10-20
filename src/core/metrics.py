import re
import statistics
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))

try:
    _LEMMATIZER = WordNetLemmatizer()
    _LEMMATIZER.lemmatize("test")
except LookupError:
    nltk.download("wordnet", quiet=True)
    _LEMMATIZER = WordNetLemmatizer()


def _clean_prefix(sentence: str) -> str:
    """Remove reference numbering prefix (e.g., 'Reference 1:') added during judge dataset creation."""
    return re.sub(r"^\s*reference\s+[\w\d]+\s*:\s*", "", sentence, flags=re.IGNORECASE)


def _preprocess(sentence: str) -> set[str]:
    cleaned = _clean_prefix(sentence)
    tokens = word_tokenize(cleaned.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens if t not in _STOP_WORDS]
    return set(tokens)


def sentence_iou(sent1: str, sent2: str) -> float:
    set1, set2 = _preprocess(sent1), _preprocess(sent2)
    union = set1 | set2
    return len(set1 & set2) / len(union) if union else 0.0


def compute_iou_stats(
    response: str, reference_chunks: list[str]
) -> tuple[float, float, float]:
    ious = [sentence_iou(response, chunk) for chunk in reference_chunks] or [0.0]
    mean = statistics.mean(ious)
    stddev = statistics.stdev(ious) if len(ious) > 1 else 0.0
    return max(ious), mean, stddev
