# macro_indices/usa.py

INDEX_CONFIG = {
    'name': 'USA',
    'series': [
        ('INDPRO',   'Production indus. USA',    3, False, 0.20),
        ('DGORDER',  'Commandes durables',       3, False, 0.15),
        ('PAYEMS',   'Emploi non-agricole',      1, False, 0.20),
        ('UNRATE',   'Taux de chômage',          1, True,  0.10),
        ('RSAFS',    'Ventes au détail',         3, False, 0.10),
        ('HOUST',    'Mises en chantier',        3, False, 0.10),
        ('UMCSENT',  'Conf. consommateur',       1, False, 0.10),
        ('CPIAUCSL', 'CPI (inflation)',          3, True,  0.05),
    ],
}