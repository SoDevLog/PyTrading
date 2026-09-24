# macro_indices/allemagne.py

INDEX_CONFIG = {
    'name': 'Allemagne',
    'series': [
        # Ticker                  Label                              lag  inv    poids
        ('DEUPROINDMISMEI',   'Production indus. Allemagne',          0,  False,  0.22),
        ('LRHUTTTTDEM156S',   'Chômage Allemagne',                    1,  True,   0.22),
        ('DEUCPIALLMINMEI',   'Inflation Allemagne',                  3,  True,   0.15),
        ('DEUSLRTTO01GYSAM',     'Ventes au détail Allemagne',           0,  False,  0.18),
        ('CSCICP02DEM460S',   'Confiance consommateur Allemagne',     1,  False,  0.23),
    ],
}