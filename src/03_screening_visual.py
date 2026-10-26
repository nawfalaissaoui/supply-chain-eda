# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# %%
INPUT_FILE = './data/supply_chain_cleaned.csv'
OUTPUT_DIR = 'figures/exploratory'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 9

# %%
def crear_directori(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print(f"Directori creat: {ruta}")
    return ruta

def guardar_grafic(nom, carpeta=OUTPUT_DIR):
    ruta_completa = f'{carpeta}/{nom}.png'
    plt.savefig(ruta_completa, bbox_inches='tight', dpi=150)
    plt.close()
    return ruta_completa

# %%
VARS_CATEGORIQUES = [
    'type', 'delivery_status', 'category_name', 'customer_city',
    'customer_country', 'customer_segment', 'customer_state',
    'department_name', 'market', 'order_city', 'order_country',
    'order_region', 'order_state', 'order_status', 'product_name',
    'shipping_mode'
]

VARS_NUMERIQUES = [
    'days_for_shipping_real', 'days_for_shipment_scheduled',
    'benefit_per_order', 'sales_per_customer', 'order_item_discount',
    'order_item_discount_rate', 'order_item_profit_ratio',
    'order_item_quantity', 'sales', 'product_price',
    'latitude_src', 'longitude_src', 'latitude_dest', 'longitude_dest'
]

VARS_BINARIES = ['late_delivery_risk']
VARS_DATES = ['order_date_dateorders', 'shipping_date_dateorders']

# %%
def realitzar_cribratge(df, vars_cat, vars_num, output_dir):
    print("INICIANT CRIBRATGE VISUAL AUTOMÀTIC")
    print("=" * 50)
    
    vars_cat = [v for v in vars_cat if v in df.columns]
    vars_num = [v for v in vars_num if v in df.columns]
    
    print(f"Variables categòriques a analitzar: {len(vars_cat)}")
    print(f"Variables numèriques a analitzar: {len(vars_num)}")
    
    total_grafics = len(vars_cat) + len(vars_num)
    print(f"Els gràfics es guardaran a: {output_dir}")
    print("=" * 50)
    
    print("\nGenerant gràfics per a variables CATEGÒRIQUES...")
    for i, col in enumerate(vars_cat, 1):
        try:
            plt.figure(figsize=(10, 5))
            
            counts = df[col].value_counts().head(20)
            sns.barplot(x=counts.values, y=counts.index, palette='viridis')
            
            plt.title(f'Cribratge: {col} (Top 20)', fontweight='bold')
            plt.xlabel('Freqüència')
            plt.tight_layout()
            
            guardar_grafic(f'cat_{col}', output_dir)
            print(f"  [{i}/{len(vars_cat)}] {col} - OK")
            
        except Exception as e:
            print(f"  [{i}/{len(vars_cat)}] Error amb {col}: {e}")
    
    print("\nGenerant gràfics per a variables NUMÈRIQUES...")
    for i, col in enumerate(vars_num, 1):
        try:
            plt.figure(figsize=(10, 5))
            
            sns.histplot(df[col].dropna(), kde=True, color='skyblue', 
                        edgecolor='black', bins=30)
            
            mean_val = df[col].mean()
            median_val = df[col].median()
            plt.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mitjana: {mean_val:.2f}', alpha=0.7)
            plt.axvline(median_val, color='green', linestyle='-.',
                       label=f'Mediana: {median_val:.2f}', alpha=0.7)
            
            plt.title(f'Cribratge: {col}', fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Freqüència')
            plt.legend()
            plt.tight_layout()
            
            guardar_grafic(f'num_{col}', output_dir)
            print(f"  [{i}/{len(vars_num)}] {col} - OK")
            
        except Exception as e:
            print(f"  [{i}/{len(vars_num)}] Error amb {col}: {e}")
    
    print("\nGenerant gràfics per a variables BINÀRIES...")
    for i, col in enumerate(VARS_BINARIES, 1):
        if col in df.columns:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                counts = df[col].value_counts()
                labels = ['No retard (0)', 'Retard (1)']
                
                bars = axes[0].bar(labels, counts.values, 
                                   color=['#27ae60', '#e74c3c'], alpha=0.8)
                axes[0].set_title(f'Distribució de {col}', fontweight='bold')
                axes[0].set_ylabel('Freqüència')
                
                for bar in bars:
                    height = bar.get_height()
                    axes[0].annotate(f'{int(height)}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                
                axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
                           colors=['#27ae60', '#e74c3c'], startangle=90,
                           explode=[0.05, 0.05])
                axes[1].set_title(f'Proporció de {col}')
                
                plt.suptitle(f'Cribratge Variable Binària: {col}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                guardar_grafic(f'bin_{col}', output_dir)
                print(f"  [{i}/{len(VARS_BINARIES)}] {col} - OK")
                
            except Exception as e:
                print(f"  [{i}/{len(VARS_BINARIES)}] Error amb {col}: {e}")
        else:
            print(f"  [{i}/{len(VARS_BINARIES)}] {col} no trobada al dataset")
    
    print("\n" + "=" * 50)
    print("CRIBRATGE COMPLETAT")
    print("=" * 50)
    print(f"Total gràfics generats: {total_grafics}")
    print(f"Ubicació: {output_dir}/")
    print("\nPISTES PER A LA REVISIÓ:")
    print("  • Distribucions asimètriques (cues llargues)")
    print("  • Concentracions extremes en categories")
    print("  • Variables constants o quasi-constants")
    print("  • Outliers evidents")
    print("=" * 50)

# %%
if __name__ == "__main__":
    
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Dataset carregat: {df.shape[0]} files × {df.shape[1]} columnes")
    except FileNotFoundError:
        print(f"ERROR: No es troba el fitxer {INPUT_FILE}")
        print("Assegura't que el dataset net existeix o modifica INPUT_FILE.")
        exit(1)
    
    crear_directori(OUTPUT_DIR)
    
    realitzar_cribratge(
        df=df,
        vars_cat=VARS_CATEGORIQUES,
        vars_num=VARS_NUMERIQUES,
        output_dir=OUTPUT_DIR
    )
    
    print("\nRESUM ESTADÍSTIC RÀPID PER VARIABLE:")
    print("-" * 50)
    for col in VARS_NUMERIQUES:
        if col in df.columns:
            print(f"{col:30s} | Mitjana: {df[col].mean():8.2f} | "
                  f"Std: {df[col].std():8.2f} | Nuls: {df[col].isnull().sum():4d}")