INDEX_CONFIG = {
    'name': 'Euro Zone',

    'series': [
        # Ticker                   Label                    lag   inv    poids
        ('PRINTO01EZQ661S',  'Production indus. ZE',        2,  False,  0.20),
        ('LRHUTTTTEZM156S',  'Chômage harmonisé ZE',        1,  True,   0.20),
        ('CP0000EZ19M086NEST','HICP Inflation ZE',          3,  True,   0.15),
        ('SLRTTO01EZQ659S',  'Ventes au détail ZE',         2,  False,  0.15),
        ('CSCICP03EZM665S',  'Confiance conso. OCDE ZE',    1,  False,  0.15),
        ('MABMM301EZM657S',  'Masse monétaire M3 ZE',       3,  False,  0.15),
    ],
}