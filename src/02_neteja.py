import pandas as pd

# 1. CARREGAR DADES
df = pd.read_csv('./data/cadena_subministrament_2015_2018.csv')
print(f"Dimensions originals: {df.shape}")

# 2. DEFINICIÓ DE VARIABLES A DESCARTAR
# Documentem aquí mateix per què les eliminem
cols_to_drop = [
    # Redundants o duplicades (tenim el nom o una altra mètrica equivalent)
    'category_id',                # Tenim category_name
    'department_id',              # Tenim department_name
    'product_card_id',            # Mateix que order_item_cardprod_id
    'product_category_id',        # Duplicada de category_id
    'order_item_product_price',   # Igual que product_price
    'order_profit_per_order',     # Tenim benefit_per_order
    'order_item_total',           # Duplicada o calculable
    'order_city_en',              # Duplicada (traducció)
    'order_state_en',             # Duplicada (traducció)
    'order_country_en',           # Duplicada (traducció)
    
    # Informació personal o irrellevant per a l'EDA (PII & soroll)
    'customer_email',
    'customer_fname',
    'customer_lname',
    'customer_password',
    'customer_zipcode',           # No útil per EDA general
    'customer_street',            # Massa granular / soroll
    'address_dest',               # Massa granular / soroll
    'product_image',              # URL tècnica
    
    # Qualitat de dada pobra o constant
    'order_zipcode',              # Molts nuls
    'product_status',             # Constant (sempre disponible)
    'order_customer_id',          # ID redundant
]

# 3. EXECUTAR LA NETEJA
print(f"\nEliminant {len(cols_to_drop)} columnes innecessàries...")
df_clean = df.drop(columns=cols_to_drop, errors='ignore') # errors='ignore' per si alguna ja no hi és


# 5. GESTIÓ DE NULS (Opcional, verificació ràpida)
# Si queden nuls a les variables importants, decidim què fer. 
# En aquest dataset, normalment les restants estan netes, però comprovem:
nuls_restants = df_clean.isnull().sum()
if nuls_restants.sum() > 0:
    print("\nAlerta: Encara queden valors nuls a:")
    print(nuls_restants[nuls_restants > 0])
    # Aquí podries afegir: df_clean = df_clean.dropna() si fos necessari

# 6. GUARDAR RESULTAT
output_file = './data/supply_chain_cleaned.csv'
df_clean.to_csv(output_file, index=False)

print("-" * 30)
print(f"PROCÉS COMPLETAT.")
print(f"Dimensions finals: {df_clean.shape}")
print(f"Dades netes guardades a: {output_file}")
print("-" * 30)

# 7. VISUALITZAR ESTRUCTURA FINAL (Per verificar tipus)
print("\nTipus de dades finals:")
print(df_clean.dtypes)