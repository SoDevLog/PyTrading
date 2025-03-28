""" Helper functions

    >pip install workalendar

    - is_date
    - datetime_now
    - working_days_in_france
    - from_date_to_days_in_past
    - format_number3f
    - format_number6f
    - format_large_number
    - format_number
    - is_empty
    - print_all_data
"""
import debug.func as debug
from workalendar.europe import France
from datetime import datetime, timedelta, date as dt_date

# ----------------------------------------------------------------------------

def is_date( str, format ):
    try:
        datetime.strptime(str, format)
        return True
    except ValueError:
        return False

# ----------------------------------------------------------------------------
# timedelta( days=1 ) to obtain data of the current day
#
def datetime_now():
    _now = datetime.now() + timedelta( days=1 )
    return datetime.strftime( _now, "%Y-%m-%d" )

# ----------------------------------------------------------------------------
# Give a date in days past
#
def datetime_past( days ):
    _now = datetime.now() - timedelta( days=days )
    return datetime.strftime( _now, "%Y-%m-%d" )

# ----------------------------------------------------------------------------
# For calulation tendency I need to know howmany open days are between
# the two dates
# 
def working_days_in_france( date_start, date_end ):
    """ Give the count of open days (business days) between the two dates """
    # To compare with the date given by holidays
    _day = dt_date( date_start.year, date_start.month, date_start.day )
    _date_end = dt_date( date_end.year, date_end.month, date_end.day )
    business_day = 0
    cal = France()
    
    # Get holidays for this year
    holidays = cal.holidays( date_start.year )
    
    # Count between the two dates
    while _day <= _date_end:
        # monday: 0 ... friday: 4
        if _day.weekday() < 5:
            found = False
            for d, name in holidays:
                if _day == d:
                    found = True
                    break
                
            if not found:
                business_day += 1
                
        _day += timedelta( days=1 ) # Go to next day
    
    return business_day

# ----------------------------------------------------------------------------

def from_date_to_days_in_past( date_start, days ):
    """ Give the date in the past from date_start couting days """
    # To compare with the date given by holidays
    date_in_past = dt_date( date_start.year, date_start.month, date_start.day )
    business_day = 0
    cal = France()
    
    # Get holidays for this year
    holidays = cal.holidays( date_start.year )
    
    # Count days in the past
    while business_day < days:
        date_in_past = date_in_past - timedelta( days=1 )
        # monday: 0 ... friday: 4
        if date_in_past.weekday() < 5:
            found = False
            for d, name in holidays:
                if date_in_past == d:
                    found = True
                    break
                
            if not found:
                business_day += 1
    
    return date_in_past

# ----------------------------------------------------------------------------

def format_number2f( num ):
    return "{:.2f}".format( num )

# ----------------------------------------------------------------------------

def format_number3f( num ):
    return "{:.3f}".format( num )

# ----------------------------------------------------------------------------

def format_number6f( num ):
    return "{:.6f}".format( num )

# ----------------------------------------------------------------------------
# The need will be to K 1000 M 1000 000 and MM 1000 M
# arg: 'pos' is there for this function to be used with FuncFormatter
#
def format_large_number(x, pos):
    """ Add an M or a K """
    if abs( x ) >= 1e6:
        s = f'{x*1e-6:.0f} M'
    elif abs( x ) >= 1e3:
        s = f'{x*1e-3:.0f} K'
    else:
        if x.is_integer():
            s = f'{x:.0f}'
        else:
            s = f'{x:.2f}'
    return s

# ----------------------------------------------------------------------------
# Format big number to be readable by user
# ----------------------------------------------------------------------------
# The idea is to display big number in format like x xxx xxx xxx
# when numer are small put decimals
#
def format_number( number, max_value=1e+12, deci=0 ):
    """ If number > max_value format to scientifique number
        else put space each 3 digits
    """    
    if abs( number ) > max_value:
        return "{:.3e}".format(number)
    
    if abs( number ) < 1000:
        return "{:.{decimals}f}".format(number, decimals=3)

    return "{:,.{decimals}f}".format(number, decimals=deci).replace(",", " ")
    
# ----------------------------------------------------------------------------
# Une fonction chargée de créer le data renvoit None
# Mais une autre peut renvoyer vide avec 0 éléments
#
def is_empty( data ):
    if data is None: 
        return True
    if len( data ) == 0:
        return True
    return False    

# ----------------------------------------------------------------------------

def print_all_data( data ):
    for _, row in data.iterrows():
        debug.print( row )