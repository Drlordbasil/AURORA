from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.

    Parameters:
    text (str): The text to analyze the sentiment of.

    Returns:
    dict: A dictionary containing the sentiment analysis results with keys "polarity" and "subjectivity".
          The "polarity" value represents the sentiment polarity ranging from -1.0 (negative) to 1.0 (positive),
          and the "subjectivity" value represents the subjectivity of the text ranging from 0.0 (objective) to 1.0 (subjective).

    Raises:
    Exception: If there is an error analyzing the sentiment.

    """
    try:
        sentiment = TextBlob(text).sentiment
        return {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}
    except Exception as e:
        raise Exception(f"Error analyzing sentiment: {e}")
