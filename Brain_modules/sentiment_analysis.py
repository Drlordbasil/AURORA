from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    """
    try:
        sentiment = TextBlob(text).sentiment
        return {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}
    except Exception as e:
        raise Exception(f"Error analyzing sentiment: {e}")
