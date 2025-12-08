"""
    - Génération du scénario
    - Exécution de la Tkinter App
"""
from smc_generateur_scenario import generate_sample_data, generate_smc_scenario
from smc_ui import SMC_Tkinter_UI

df = generate_sample_data()
#df, _ = generate_smc_scenario( lg_data=150, start_price=100.0, seed=42 )
#ui = SMC_Tkinter_UI( df )
ui = SMC_Tkinter_UI( df, generate_sample_data )
ui.run_smc() # create SMC_Engine apply parameters
ui.plot() # afficher un premier graphique
ui.run()

