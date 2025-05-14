""" 
    Gros souci avec pymeeus

    >python test_workalendar.py

"""

from workalendar.europe import France

cal = France()
print( cal.holidays(2025) )

# resultat attendu :
#    [(datetime.date(2025, 1, 1), 'New year'), (datetime.date(2025, 4, 21), 'Easter Monday'), (datetime.date(2025, 5, 1), 'Labour Day'), (datetime.date(2025, 5, 8), 'Victory in Europe Day'), (datetime.date(2025, 5, 29), 'Ascension Thursday'), (datetime.date(2025, 6, 9), 'Whit Monday'), (datetime.date(2025, 7, 14), 'Bastille Day'), (datetime.date(2025, 8, 15), 'Assumption of Mary to Heaven'), (datetime.date(2025, 11, 1), 'All Saints Day'), (datetime.date(2025, 11, 11), 'Armistice Day'), (datetime.date(2025, 12, 25), 'Christmas Day')]