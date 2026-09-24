# macro_indices/france.py

INDEX_CONFIG = {
    'name': 'France',
    'series': [
        # Ticker                  Label                         lag  inv    poids
        ('FRAPRINTO01GYSAM',  'Production indus. France',       0,  False,  0.22),
        ('LRHUTTTTFRM156S',   'Chômage France',                 1,  True,   0.22),
        ('CP0000FRM086NEST',  'Inflation HICP France',          3,  True,   0.15),
        ('FRASLRTTO01GYSAM',  'Ventes au détail France',        0,  False,  0.18),
        ('CSCICP02FRM460S',   'Confiance consommateur France',        1,  False,  0.23),
    ],
}