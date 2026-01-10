def fair_odds(prob):
    return round(1/prob,2)

def value_percentage(book_odds, fair_odds):
    return round((book_odds/fair_odds - 1)*100,2)
