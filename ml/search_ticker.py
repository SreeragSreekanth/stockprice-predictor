from yahooquery import search

def get_ticker_from_name(name):
    """
    Returns the first Yahoo Finance ticker matching the company name.
    """
    try:
        results = search(name)
        quotes = results.get('quotes', [])
        if quotes:
            return quotes[0]['symbol']
    except:
        return None
