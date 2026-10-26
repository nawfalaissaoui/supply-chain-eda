"""
================================================================================
PRÀCTICA 1 - ANÀLISI EXPLORATÒRIA DE DADES
El poder de les dades: Anàlisi d'una cadena de subministrament simple
================================================================================
Sistemes d'Informació en les Organitzacions (2025/26)
Universitat Rovira i Virgili

Autors: Nawfal Aissaoui
Data: Octubre 2025
================================================================================

ESTRUCTURA DE L'ANÀLISI:
------------------------
1. Càrrega i preparació de dades
2. Anàlisi univariant (descripció individual de variables)
3. Anàlisi bivariant (relacions entre parells de variables)
4. Anàlisi multivariant (interaccions complexes)
5. Estudis específics originals (insights de negoci)

================================================================================
"""
# %% ============================================================================
# IMPORTS I CONFIGURACIÓ GLOBAL
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kstest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Configuració estètica
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 100
sns.set_palette("husl")

# Directoris per organitzar outputs
BASE_DIR = 'figures/final'
DIRS = {
    'univariant_cat': f'{BASE_DIR}/01_univariant/categoriques',
    'univariant_num': f'{BASE_DIR}/01_univariant/numeriques',
    'bivariant_catnum': f'{BASE_DIR}/02_bivariant/cat_vs_num',
    'bivariant_numnum': f'{BASE_DIR}/02_bivariant/num_vs_num',
    'bivariant_catcat': f'{BASE_DIR}/02_bivariant/cat_vs_cat',
    'multivariant': f'{BASE_DIR}/03_multivariant',
    'estudis': f'{BASE_DIR}/04_estudis_especifics',
    'reports': 'reports'
}

for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

def save_fig(name, category):
    """Guarda figura en la carpeta corresponent"""
    plt.savefig(f'{DIRS[category]}/{name}.png', bbox_inches='tight', dpi=150)
    plt.close()

def print_section(title):
    """Imprimeix capçalera de secció"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

# %% ============================================================================
# 1. CÀRREGA I PREPARACIÓ DE DADES
# ==============================================================================

print_section("1. CÀRREGA I PREPARACIÓ DE DADES")

df = pd.read_csv('./data/supply_chain_cleaned.csv')
print(f"Dataset carregat: {df.shape[0]:,} registres × {df.shape[1]} variables")

# Conversió de dates
date_cols = ['order_date_dateorders', 'shipping_date_dateorders']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Variables temporals derivades
df['order_year'] = df['order_date_dateorders'].dt.year
df['order_month'] = df['order_date_dateorders'].dt.month
df['order_quarter'] = df['order_date_dateorders'].dt.quarter
df['order_day_name'] = df['order_date_dateorders'].dt.day_name()
df['order_week'] = df['order_date_dateorders'].dt.isocalendar().week
df['order_year_month'] = df['order_date_dateorders'].dt.to_period('M')

# Variables derivades clau
df['delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']
df['is_profitable'] = (df['benefit_per_order'] > 0).astype(int)
df['discount_amount'] = df['product_price'] * df['order_item_discount_rate']
df['final_price'] = df['product_price'] - df['discount_amount']

print(f"Període: {df['order_date_dateorders'].min().date()} → {df['order_date_dateorders'].max().date()}")

# %% ============================================================================
# 2. CLASSIFICACIÓ DE VARIABLES
# ==============================================================================

print_section("2. CLASSIFICACIÓ DE VARIABLES")

VARS_CATEGORIQUES = {
    'critica': ['delivery_status'],
    'alta': ['customer_segment', 'category_name', 'market', 'order_country', 
             'order_region', 'shipping_mode', 'order_status'],
    'mitjana': ['customer_country', 'customer_state', 'department_name', 'product_name', 'type']
}

VARS_NUMERIQUES = {
    'critica': ['days_for_shipping_real', 'days_for_shipment_scheduled',
                'benefit_per_order', 'order_item_profit_ratio', 'sales'],
    'alta': ['product_price', 'order_item_quantity', 'order_item_discount_rate'],
    'derivades': ['delay_days', 'discount_amount', 'final_price']
}

all_cat_vars = sum(VARS_CATEGORIQUES.values(), [])
all_num_vars = sum(VARS_NUMERIQUES.values(), [])

print(f"Variables categòriques: {len(all_cat_vars)}")
print(f"Variables numèriques: {len(all_num_vars)}")
# %% ============================================================================
# 3. ANÀLISI UNIVARIANT - VARIABLES CATEGÒRIQUES
# ==============================================================================

print_section("3.1 ANÀLISI UNIVARIANT - CATEGÒRIQUES")

def analitzar_categorica(col, top_n=15):
    """
    Analitza una variable categòrica amb estadístics i un gràfic de barres net.
    """
    print(f"\n{'─'*80}\n {col.upper()}\n{'─'*80}")

    try:
        n_total = len(df)
        freq = df[col].value_counts()
        n_unique = df[col].nunique()

        print(f"Categories úniques: {n_unique:,}")
        print(f"Dominant: '{freq.index[0]}' ({freq.iloc[0]/n_total*100:.1f}%)")
        
        # Top N per visualitzar
        top_data = freq.head(top_n)
        
        # Configuració del gràfic (Només 1 ax, més net)
        plt.figure(figsize=(10, max(6, len(top_data) * 0.4))) # Altura dinàmica
        
        # Barplot horitzontal
        ax = sns.barplot(x=top_data.values, y=top_data.index, palette='viridis')
        
        # Títol i etiquetes
        plt.title(f'Distribució de {col} (Top {top_n})', fontweight='bold', fontsize=14)
        plt.xlabel('Freqüència')
        
        # Afegir etiquetes de dades al final de les barres (Valor absolut | %)
        for i, v in enumerate(top_data.values):
            pct = v / n_total * 100
            ax.text(v + (v * 0.01), i, f' {v:,} ({pct:.1f}%)', va='center', fontweight='bold', fontsize=9)
        
        # Ajustar marges
        plt.tight_layout()
        save_fig(f'cat_{col}', 'univariant_cat')
        print(f"Gràfic guardat: cat_{col}.png")
        
    except Exception as e:
        print(f"Error analitzant {col}: {e}")

print("\nAnalitzant variables categòriques prioritàries...")

for priority in ['critica', 'alta']:
    for var in VARS_CATEGORIQUES[priority]:
        if var in df.columns:
            analitzar_categorica(var, top_n=20 if var == 'category_name' else 15)

# %% ============================================================================
# 4. ANÀLISI UNIVARIANT - VARIABLES NUMÈRIQUES
# ==============================================================================

print_section("3.2 ANÀLISI UNIVARIANT - NUMÈRIQUES")

def analitzar_numerica(col, bins=50):
    """
        Analitza una variable numèrica amb enfocament en distribucions financeres/logístiques.
        Inclou visió logarítmica per gestionar outliers.
    """
    print(f"\n{'─'*80}\n{col.upper()}\n{'─'*80}")
    
    try:
        data = df[col].dropna()
        if len(data) == 0:
            print("Variable buida.")
            return

        desc = data.describe()
        
        # Estadístics Avançats
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        print(f"ESTADÍSTICS CLAU:")
        print(f"   - Mitjana:   {desc['mean']:,.2f}")
        print(f"   - Mediana:   {desc['50%']:,.2f}")
        print(f"   - Desv. Std: {desc['std']:,.2f}")
        print(f"   - Skewness:  {skewness:.2f} {'(Asimètrica)' if abs(skewness)>1 else '(Simètrica)'}")
        print(f"   - Kurtosis:  {kurtosis:.2f}")
        
        # Detecció d'Outliers (Mètode IQR)
        Q1, Q3 = desc['25%'], desc['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        print(f"  • Outliers:  {len(outliers):,} ({len(outliers)/len(data)*100:.2f}%)")

        # Visualització (3 subplots: Histograma, Boxplot, Histograma Logarítmic)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Histograma + KDE (Escala Lineal)
        sns.histplot(data, bins=bins, kde=True, color='skyblue', edgecolor='black', ax=axes[0])
        axes[0].axvline(data.mean(), color='red', linestyle='--', label='Mitjana')
        axes[0].axvline(data.median(), color='green', linestyle='-', label='Mediana')
        axes[0].set_title(f'{col} (Lineal)', fontweight='bold')
        axes[0].legend()
        
        # 2. Boxplot (Horitzontal per llegir millor)
        sns.boxplot(x=data, ax=axes[1], color='lightcoral', flierprops={"marker": "x"})
        axes[1].set_title(f'{col} (Boxplot)', fontweight='bold')
        axes[1].set_xlabel('')
        
        # 3. Histograma (Escala Logarítmica) - CLAU PER A DATA SCIENCE
        # Només si tenim valors positius per evitar errors amb log
        if (data > 0).all():
            sns.histplot(data, bins=bins, kde=True, log_scale=True, color='teal', edgecolor='black', ax=axes[2])
            axes[2].set_title(f'{col} (Log Scale - Per veure cues)', fontweight='bold')
        else:
            # Si hi ha negatius (ex: beneficis), fem un Violin Plot en lloc de Log
            sns.violinplot(x=data, ax=axes[2], color='teal')
            axes[2].set_title(f'{col} (Violin Plot - Densitat)', fontweight='bold')

        plt.tight_layout()
        save_fig(f'num_{col}', 'univariant_num')
        print(f"Gràfic guardat: num_{col}.png")

    except Exception as e:
        print(f"Error analitzant {col}: {e}")

print("\nAnalitzant variables numèriques prioritàries...")

for priority in ['critica', 'alta', 'derivades']: 
    if priority in VARS_NUMERIQUES: # Per seguretat
        for var in VARS_NUMERIQUES[priority]:
            if var in df.columns:
                analitzar_numerica(var)
# %% ============================================================================
# 5. ANÀLISI BIVARIANT
# ==============================================================================

print_section("4. ANÀLISI BIVARIANT")

# ------------------------------------------------------------------------------
# 5.1 RELACIÓ CATEGÒRICA VS NUMÈRICA (Amb detall estadístic)
# ------------------------------------------------------------------------------

def analitzar_cat_vs_num(cat_col, num_col, top_n=10):
    print(f"\n{'─'*80}\n{cat_col.upper()} × {num_col.upper()}\n{'─'*80}")
    
    try:
        # Filtrar Top N
        top_cats = df[cat_col].value_counts().head(top_n).index
        data_filtered = df[df[cat_col].isin(top_cats)]
        
        # 1. TAULA ESTADÍSTICA
        grouped = data_filtered.groupby(cat_col)[num_col].agg([
            ('n', 'count'), ('mitjana', 'mean'), ('mediana', 'median'), ('std', 'std')
        ]).round(2)
        print(grouped.sort_values('mitjana', ascending=False))
        
        # 2. TEST ESTADÍSTIC (Kruskal-Wallis)
        groups = [data_filtered[data_filtered[cat_col] == cat][num_col].dropna() for cat in top_cats]
        stat, pval = stats.kruskal(*groups)
        print(f"\nSignificança (K-W): p-value={pval:.4e} {'(Significatiu)' if pval < 0.05 else '(No sign.)'}")
        
        # 3. GRÀFIC DOBLE (Boxplot + Mitjanes amb error)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot esquerra
        order = data_filtered.groupby(cat_col)[num_col].median().sort_values(ascending=False).index
        sns.boxplot(data=data_filtered, x=cat_col, y=num_col, order=order, palette='viridis', showfliers=False, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_title(f'Distribució: {num_col} per {cat_col}', fontweight='bold')
        
        # Barplot dreta (Mitjanes + Desviació Estàndard)
        means = grouped['mitjana'].sort_values(ascending=False)
        stds = grouped['std'].reindex(means.index)
        ax2.barh(range(len(means)), means.values, color='teal', alpha=0.6)
        ax2.errorbar(means.values, range(len(means)), xerr=stds.values, fmt='none', ecolor='red', capsize=5)
        ax2.set_yticks(range(len(means)))
        ax2.set_yticklabels(means.index)
        ax2.set_title('Mitjanes i Desviació Estàndard', fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        save_fig(f'{cat_col}_vs_{num_col}', 'bivariant_catnum')
        print(f"Gràfic guardat: {cat_col}_vs_{num_col}.png")

    except Exception as e:
        print(f"Error: {e}")

# Execució dels creuaments més interessants per al negoci
analitzar_cat_vs_num('shipping_mode', 'delay_days', top_n=4)       # Clau per logística
analitzar_cat_vs_num('market', 'sales', top_n=5)                   # Clau per vendes
analitzar_cat_vs_num('category_name', 'benefit_per_order', top_n=10) # Clau per rendibilitat
analitzar_cat_vs_num('customer_segment', 'benefit_per_order', top_n=3)

# ------------------------------------------------------------------------------
# 5.2 RELACIÓ CATEGÒRICA VS CATEGÒRICA (Anàlisi de Risc)
# ------------------------------------------------------------------------------
# Aquesta funció és CRÍTICA per objectiu de "Late Delivery Risk"

def analitzar_risc_retard(cat_col):
    print(f"\n{'─'*80}\nRISC DE RETARD per {cat_col.upper()}\n{'─'*80}")
    
    try:
        # Calculem el % de risc (mitjana de la variable binària 0/1)
        risk = df.groupby(cat_col)['late_delivery_risk'].mean().sort_values(ascending=False) * 100
        
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x=risk.index, y=risk.values, palette='magma')
        
        plt.title(f'Probabilitat de Retard segons {cat_col}', fontweight='bold')
        plt.ylabel('% de Comandes amb Retard')
        plt.ylim(0, 100) # Fixem l'escala per veure la gravetat real
        plt.xticks(rotation=45, ha='right')
        
        # Posem el % a sobre de la barra
        for i, v in enumerate(risk.values):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
            
        plt.tight_layout()
        save_fig(f'risk_{cat_col}', 'bivariant_catcat')
        print(f"Gràfic guardat: risk_{cat_col}.png")
        
    except Exception as e:
        print(f"Error: {e}")

# Execució de riscos
analitzar_risc_retard('shipping_mode') # El més important!
analitzar_risc_retard('market')
analitzar_risc_retard('customer_segment')

# ------------------------------------------------------------------------------
# 5.3 RELACIÓ NUMÈRICA VS NUMÈRICA (Correlacions)
# ------------------------------------------------------------------------------

print(f"\n{'─'*80}\nMATRIU DE CORRELACIÓ\n{'─'*80}")

vars_corr = ['benefit_per_order', 'sales', 'product_price', 
             'order_item_quantity', 'order_item_discount', 'delay_days', 
             'days_for_shipping_real']

# Filtrem només les que existeixen
vars_existents = [v for v in vars_corr if v in df.columns]
corr_matrix = df[vars_existents].corr(method='pearson')

# 1. Imprimir correlacions fortes
print("\nCORRELACIONS FORTES (|r| > 0.5):")
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.5:
            print(f"  {corr_matrix.index[i]:25s} ↔ {corr_matrix.columns[j]:25s}  r={r:+.3f}")

# 2. Heatmap Visual
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlacions entre variables clau', fontweight='bold')
plt.tight_layout()
save_fig('correlation_matrix', 'bivariant_numnum')

# 3. Scatterplot Vendes vs Benefici (MOLT IMPORTANT PER A NEGOCI)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sales', y='benefit_per_order', 
                hue='delivery_status', alpha=0.6, palette='deep')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Relació: Vendes vs Benefici', fontweight='bold')
plt.tight_layout()
save_fig('scatter_sales_benefit', 'bivariant_numnum')
print("Scatterplot guardat.")

# %% ============================================================================
# 6. ANÀLISI MULTIVARIANT (3+ Variables)
# ==============================================================================

print_section("5. ANÀLISI MULTIVARIANT")

# ------------------------------------------------------------------------------
# 6.1 MAPA DE CALOR: RENDIBILITAT (Mercat x Categoria)
# ------------------------------------------------------------------------------
# Objectiu: Veure on guanyem diners realment.
print("Generant Heatmap de Rendibilitat...")

try:
    # 1. Preparem dades: Pivota Market vs Category amb el Benefici com a valor
    # Filtrem només les Top 10 categories per no fer un mapa gegant
    top_cats = df['category_name'].value_counts().head(10).index
    df_top = df[df['category_name'].isin(top_cats)]
    
    heatmap_data = df_top.pivot_table(index='category_name', 
                                      columns='market', 
                                      values='benefit_per_order', 
                                      aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='RdYlGn', center=0, linewidths=.5)
    plt.title('Benefici Mitjà ($) per Categoria i Mercat', fontweight='bold')
    plt.tight_layout()
    save_fig('heatmap_benefit_market_cat', 'multivariant')
    print("Gràfic guardat: heatmap_benefit_market_cat.png")

except Exception as e:
    print(f"Error Heatmap: {e}")

# ------------------------------------------------------------------------------
# 6.2 EL MAPA DEL DELICTE: Retards per Regió i Mode
# ------------------------------------------------------------------------------
# Objectiu: Veure si el problema de "Second Class" és global o local.
print("Generant Anàlisi de Retards per Regió...")

try:
    # Filtrem les regions amb més volum per no saturar
    top_regions = df['order_region'].value_counts().head(8).index
    df_reg = df[df['order_region'].isin(top_regions)]
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_reg, x='order_region', y='delay_days', 
                hue='shipping_mode', palette='magma', errorbar=None)
    
    plt.title('Dies de Retard Mitjans per Regió i Mode d\'Enviament (Top 8 Regions)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    save_fig('bar_delay_region_mode', 'multivariant')
    print("Gràfic guardat: bar_delay_region_mode.png")

except Exception as e:
    print(f"Error Regions: {e}")

# ------------------------------------------------------------------------------
# 6.3 INTERACCIÓ: Descompte vs Benefici per Regió
# ------------------------------------------------------------------------------
# Objectiu: Veure si els descomptes fan més mal en algunes regions que en altres.
print("Generant Scatterplot Descomptes...")

try:
    # Fem un sample si hi ha massa punts, per fer-ho llegible
    df_sample = df.sample(n=5000, random_state=42)
    
    g = sns.lmplot(data=df_sample, x='order_item_discount_rate', y='benefit_per_order', 
                   hue='market', height=6, aspect=1.5, scatter_kws={'alpha':0.3})
    
    g.set_axis_labels("Taxa de Descompte", "Benefici per Comanda ($)")
    plt.title('Impacte del Descompte en el Benefici per Mercat', fontweight='bold')
    plt.tight_layout()
    # lmplot guarda diferent, usem el mètode savefig de matplotlib directament sobre la figura actual
    plt.savefig(f'{DIRS["multivariant"]}/scatter_discount_benefit_market.png', dpi=150)
    plt.close()
    print("Gràfic guardat: scatter_discount_benefit_market.png")

except Exception as e:
    print(f"Error Scatter: {e}")
# %%
# %% ============================================================================
# 7. ESTUDIS ESPECÍFICS D'APROFUNDIMENT
# ==============================================================================

print_section("6. ESTUDIS ESPECÍFICS D'APROFUNDIMENT")

# ==============================================================================
# ESTUDI 1: ANÀLISI DE PARETO DINÀMICA - LA CONCENTRACIÓ DEL RISC
# ==============================================================================
"""
OBJECTIU: Identificar la dependència extrema del negoci en pocs productes/clients
TÈCNICA: Corba de Pareto amb descomposició temporal
ORIGINALITAT: No només fem el 80/20 clàssic, sinó que analitzem l'evolució
             de la concentració al llarg del temps per detectar vulnerabilitats
"""

print("\n" + "="*80)
print("ESTUDI 1: ANÀLISI DE PARETO DINÀMICA")
print("="*80)

try:
    # 1.1 Pareto de Productes (Anàlisi Global)
    product_sales = df.groupby('product_name')['sales'].sum().sort_values(ascending=False)
    cumulative_pct = (product_sales.cumsum() / product_sales.sum() * 100)
    
    # Trobar el punt del 80%
    pareto_80 = cumulative_pct[cumulative_pct <= 80].index
    pareto_90 = cumulative_pct[cumulative_pct <= 90].index
    
    print(f"\nCONCENTRACIÓ DE VENDES:")
    print(f"   - Total productes únics: {len(product_sales):,}")
    print(f"   - Productes que generen 80% de vendes: {len(pareto_80)} ({len(pareto_80)/len(product_sales)*100:.1f}%)")
    print(f"   - Productes que generen 90% de vendes: {len(pareto_90)} ({len(pareto_90)/len(product_sales)*100:.1f}%)")
    print(f"   - Ràtio de concentració: {len(product_sales) / len(pareto_80):.1f}:1")
    
    # Top 5 productes
    print(f"\nTOP 5 PRODUCTES (BLOCKBUSTERS):")
    for i, (prod, sales_val) in enumerate(product_sales.head(5).items(), 1):
        pct = sales_val / product_sales.sum() * 100
        print(f"   {i}. {prod[:50]}: ${sales_val:,.0f} ({pct:.1f}%)")
    
    # Visualització de la corba de Pareto
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gràfic 1: Corba de Pareto completa
    x_values = range(1, len(cumulative_pct) + 1)
    ax1.plot(x_values, cumulative_pct.values, color='#2E86AB', linewidth=2.5)
    ax1.axhline(80, color='red', linestyle='--', linewidth=1.5, label='Regla 80/20')
    ax1.axvline(len(pareto_80), color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.fill_between(x_values, 0, cumulative_pct.values, alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Nombre de Productes (ordenats per vendes)', fontweight='bold')
    ax1.set_ylabel('% Acumulat de Vendes', fontweight='bold')
    ax1.set_title('Corba de Pareto: Concentració de Vendes', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.annotate(f'{len(pareto_80)} productes\n= 80% vendes', 
                 xy=(len(pareto_80), 80), xytext=(len(pareto_80)*2, 70),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, fontweight='bold', color='red')
    
    # Gràfic 2: Top 20 productes amb barres
    top20 = product_sales.head(20)
    colors = ['#A23B72' if i < len(pareto_80) else '#C9ADA7' for i in range(len(top20))]
    ax2.barh(range(len(top20)), top20.values, color=colors)
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels([name[:30] for name in top20.index], fontsize=8)
    ax2.set_xlabel('Vendes Totals ($)', fontweight='bold')
    ax2.set_title('Top 20 Productes per Vendes', fontweight='bold', fontsize=14)
    ax2.invert_yaxis()
    
    # Afegir percentatges
    for i, (val, pct) in enumerate(zip(top20.values, top20.values/product_sales.sum()*100)):
        ax2.text(val, i, f' {pct:.1f}%', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    save_fig('pareto_products', 'estudis')
    print("\nGràfic guardat: pareto_products.png")
    
    # 1.2 Evolució temporal de la concentració (INNOVACIÓ)
    print(f"\nEVOLUCIÓ DE LA CONCENTRACIÓ (Anàlisi de Risc Temporal):")
    
    concentration_by_year = []
    for year in sorted(df['order_year'].unique()):
        df_year = df[df['order_year'] == year]
        year_sales = df_year.groupby('product_name')['sales'].sum().sort_values(ascending=False)
        year_cumsum = (year_sales.cumsum() / year_sales.sum() * 100)
        n_products_80 = len(year_cumsum[year_cumsum <= 80])
        concentration_ratio = len(year_sales) / n_products_80 if n_products_80 > 0 else 0
        concentration_by_year.append({
            'year': year,
            'total_products': len(year_sales),
            'products_80pct': n_products_80,
            'concentration_ratio': concentration_ratio
        })
        print(f"   {year}: {n_products_80} productes generen 80% (Ràtio: {concentration_ratio:.1f}:1)")
    
    conc_df = pd.DataFrame(concentration_by_year)
    
    # Visualització de l'evolució
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(conc_df['year'], conc_df['concentration_ratio'], marker='o', 
            linewidth=2.5, markersize=10, color='#E63946')
    ax.fill_between(conc_df['year'], 0, conc_df['concentration_ratio'], alpha=0.3, color='#E63946')
    ax.set_xlabel('Any', fontweight='bold')
    ax.set_ylabel('Ràtio de Concentració (Total:Top80%)', fontweight='bold')
    ax.set_title('Evolució de la Dependència en Productes Estrella (2015-2018)', 
                 fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)
    
    # Destacar tendència
    if conc_df['concentration_ratio'].iloc[-1] > conc_df['concentration_ratio'].iloc[0]:
        trend_text = "AUGMENT del risc\n(Més dependència)"
        trend_color = 'red'
    else:
        trend_text = "REDUCCIÓ del risc\n(Més diversificació)"
        trend_color = 'green'
    
    ax.text(0.05, 0.95, trend_text, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', color=trend_color,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_fig('pareto_evolution', 'estudis')
    print("Gràfic guardat: pareto_evolution.png")
    
except Exception as e:
    print(f"Error en Estudi 1: {e}")

# ==============================================================================
# ESTUDI 2: SEGMENTACIÓ AVANÇADA DE CLIENTS (RFM + K-MEANS)
# ==============================================================================
"""
OBJECTIU: Identificar perfils de clients amb comportaments diferents
TÈCNICA: Anàlisi RFM (Recency, Frequency, Monetary) + Clustering K-Means
ORIGINALITAT: Combinació de màrqueting analític amb machine learning no supervisat
             per crear segments accionables
"""

print("\n" + "="*80)
print("ESTUDI 2: SEGMENTACIÓ INTEL·LIGENT DE CLIENTS (RFM + CLUSTERING)")
print("="*80)

try:
    # 2.1 Càlcul de mètriques RFM
    print("\nCALCULANT MÈTRIQUES RFM...")
    
    # Data de referència (última compra del dataset)
    reference_date = df['order_date_dateorders'].max()
    
    # Agreguem per customer_id
    rfm = df.groupby('customer_id').agg({
        'order_date_dateorders': lambda x: (reference_date - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'sales': 'sum'  # Monetary
    }).rename(columns={
        'order_date_dateorders': 'Recency',
        'order_id': 'Frequency',
        'sales': 'Monetary'
    })
    
    print(f"   - Clients analitzats: {len(rfm):,}")
    print(f"\nSTADÍSTICS RFM:")
    print(rfm.describe().round(2))
    
    # 2.2 Normalització per K-Means
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # 2.3 Determinació del nombre òptim de clusters (Mètode del Colze)
    print(f"\nDETERMINANT NOMBRE ÒPTIM DE CLUSTERS...")
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)
    
    # Mètode del colze
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='#06A77D')
    ax1.set_xlabel('Nombre de Clusters (K)', fontweight='bold')
    ax1.set_ylabel('Inèrcia (Suma de distàncies²)', fontweight='bold')
    ax1.set_title('Mètode del Colze per K-Means', fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.annotate('Colze òptim', xy=(4, inertias[2]), xytext=(6, inertias[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    # 2.4 Clustering amb K=4 (segons el colze)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    print(f"Clustering completat amb K={optimal_k}")
    
    # 2.5 Caracterització dels Clusters
    print(f"\nPERFILS DE CLIENTS IDENTIFICATS:")
    
    cluster_profiles = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)
    
    cluster_profiles['Size'] = rfm.groupby('Cluster').size()
    cluster_profiles['Size_pct'] = (cluster_profiles['Size'] / len(rfm) * 100).round(1)
    
    # Assignar noms descriptius
    cluster_names = {
        cluster_profiles['Monetary'].idxmax(): 'VIP (Alt Valor)',
        cluster_profiles['Frequency'].idxmax(): 'Fidels (Alta Freqüència)',
        cluster_profiles['Recency'].idxmax(): 'Dormits (Inactius)',
        cluster_profiles[cluster_profiles['Monetary'] < cluster_profiles['Monetary'].median()].index[0]: 'Ocasionals (Baix Valor)'
    }
    
    # Reordenar per claritat
    for cluster_id in cluster_profiles.index:
        if cluster_id not in cluster_names:
            cluster_names[cluster_id] = f'Segment {cluster_id}'
    
    print(cluster_profiles)
    print("\nETIQUETES DELS SEGMENTS:")
    for cluster_id, name in cluster_names.items():
        size = cluster_profiles.loc[cluster_id, 'Size']
        pct = cluster_profiles.loc[cluster_id, 'Size_pct']
        print(f"   Cluster {cluster_id}: {name} ({size:,} clients, {pct}%)")
    
    # 2.6 Visualització dels Clusters
    # Scatter 3D simulat amb 2 projeccions
    ax2 = plt.subplot(122)
    scatter = ax2.scatter(rfm['Frequency'], rfm['Monetary'], 
                         c=rfm['Cluster'], cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.set_xlabel('Frequency (Nombre de Compres)', fontweight='bold')
    ax2.set_ylabel('Monetary (Valor Total $)', fontweight='bold')
    ax2.set_title('Distribució dels Segments de Clients', fontweight='bold')
    
    # Afegir centroides
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, centroid in enumerate(centroids_original):
        ax2.scatter(centroid[1], centroid[2], marker='X', s=300, 
                   edgecolors='red', linewidths=2, label=f'Centroide {i}')
    
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')
    plt.tight_layout()
    save_fig('rfm_clustering', 'estudis')
    print("\nGràfic guardat: rfm_clustering.png")
    
    # 2.7 Boxplots per variable RFM
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, metric in enumerate(['Recency', 'Frequency', 'Monetary']):
        sns.boxplot(data=rfm, x='Cluster', y=metric, palette='Set2', ax=axes[idx])
        axes[idx].set_title(f'Distribució de {metric} per Segment', fontweight='bold')
        axes[idx].set_xlabel('Cluster')
    
    plt.tight_layout()
    save_fig('rfm_boxplots', 'estudis')
    print("Gràfic guardat: rfm_boxplots.png")
    
except Exception as e:
    print(f"Error en Estudi 2: {e}")

# ==============================================================================
# ESTUDI 3: ANÀLISI DE RENDIBILITAT AMB BREAK-EVEN (THRESHOLD ANALYSIS)
# ==============================================================================
"""
OBJECTIU: Determinar el punt crític on els descomptes destrueixen valor
TÈCNICA: Anàlisi de punt mort (break-even) amb simulació d'escenaris
ORIGINALITAT: No només mostrem correlació, sinó que calculem el límit operatiu
             real per a la política comercial
"""

print("\n" + "="*80)
print("ESTUDI 3: ANÀLISI DEL PUNT DE RUPTURA (BREAK-EVEN ANALYSIS)")
print("="*80)

try:
    # 3.1 Segmentació per nivells de descompte
    print("\nANALITZANT IMPACTE DELS DESCOMPTES EN LA RENDIBILITAT...")
    
    # Crear bins de descompte
    df['discount_bin'] = pd.cut(df['order_item_discount_rate'], 
                                bins=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.0],
                                labels=['0-5%', '5-10%', '10-15%', '15-20%', 
                                       '20-25%', '25-30%', '>30%'])
    
    # Agregar per bin
    discount_analysis = df.groupby('discount_bin', observed=True).agg({
        'benefit_per_order': ['mean', 'median', 'std', 'count'],
        'sales': 'mean',
        'order_item_profit_ratio': 'mean'
    }).round(2)
    
    discount_analysis.columns = ['_'.join(col).strip() for col in discount_analysis.columns.values]
    discount_analysis = discount_analysis.rename(columns={
        'benefit_per_order_mean': 'Benefici_Mitjà',
        'benefit_per_order_median': 'Benefici_Mediana',
        'benefit_per_order_std': 'Desviació',
        'benefit_per_order_count': 'N_Comandes',
        'sales_mean': 'Vendes_Mitjanes',
        'order_item_profit_ratio_mean': 'Marge_Mitjà'
    })
    
    print("\nRENDIBILITAT PER TRAM DE DESCOMPTE:")
    print(discount_analysis)
    
    # Trobar el break-even point
    positive_benefit = discount_analysis[discount_analysis['Benefici_Mitjà'] > 0]
    if len(positive_benefit) > 0:
        last_profitable = positive_benefit.index[-1]
        print(f"\nPUNT CRÍTIC DETECTAT:")
        print(f"   - Últim tram rendible: {last_profitable}")
        print(f"   - Benefici en aquest tram: ${positive_benefit.loc[last_profitable, 'Benefici_Mitjà']:.2f}")
    
    # 3.2 Visualització amb zona de perill
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gràfic 1: Benefici mitjà amb barres de error
    x_pos = range(len(discount_analysis))
    colors = ['#06A77D' if val > 0 else '#E63946' for val in discount_analysis['Benefici_Mitjà']]
    
    bars = ax1.bar(x_pos, discount_analysis['Benefici_Mitjà'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.errorbar(x_pos, discount_analysis['Benefici_Mitjà'], 
                yerr=discount_analysis['Desviació'], fmt='none', 
                ecolor='black', capsize=5, alpha=0.6)
    
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Punt Mort (Break-even)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(discount_analysis.index, rotation=0)
    ax1.set_xlabel('Tram de Descompte', fontweight='bold')
    ax1.set_ylabel('Benefici Mitjà per Comanda ($)', fontweight='bold')
    ax1.set_title('Impacte del Descompte en la Rendibilitat', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Anotar zona de pèrdues
    if len(discount_analysis[discount_analysis['Benefici_Mitjà'] < 0]) > 0:
        first_loss = discount_analysis[discount_analysis['Benefici_Mitjà'] < 0].index[0]
        loss_idx = list(discount_analysis.index).index(first_loss)
        ax1.axvspan(loss_idx - 0.5, len(x_pos) - 0.5, alpha=0.2, color='red', 
                   label='Zona de Pèrdues')
        ax1.text(loss_idx + 0.5, ax1.get_ylim()[1] * 0.9, 
                'ZONA DE RISC', fontsize=12, fontweight='bold', 
                color='red', ha='left')
    
    # Gràfic 2: Marge de benefici (profit ratio)
    ax2.plot(x_pos, discount_analysis['Marge_Mitjà'] * 100, 
            marker='o', linewidth=2.5, markersize=10, color='#F18F01')
    ax2.fill_between(x_pos, 0, discount_analysis['Marge_Mitjà'] * 100, 
                     alpha=0.3, color='#F18F01')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(discount_analysis.index, rotation=0)
    ax2.set_xlabel('Tram de Descompte', fontweight='bold')
    ax2.set_ylabel('Marge de Benefici (%)', fontweight='bold')
    ax2.set_title('Erosió del Marge per Descompte', fontweight='bold', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_fig('breakeven_analysis', 'estudis')
    print("\nGràfic guardat: breakeven_analysis.png")
    
    # 3.3 Simulació d'escenari: "Què passaria si...?"
    print(f"\nSIMULACIÓ D'ESCENARI - POLÍTICA DE DESCOMPTE LIMITADA:")
    
    # Escenari 1: Límit del 12%
    scenario_12 = df[df['order_item_discount_rate'] <= 0.12]
    current_profit = df['benefit_per_order'].sum()
    scenario_profit = scenario_12['benefit_per_order'].sum()
    lost_sales = df['sales'].sum() - scenario_12['sales'].sum()
    
    print(f"\n   ESCENARI: Prohibir descomptes > 12%")
    print(f"   - Benefici actual:        ${current_profit:,.0f}")
    print(f"   - Benefici amb límit 12%: ${scenario_profit:,.0f}")
    print(f"   - Diferència:             ${scenario_profit - current_profit:,.0f} ({(scenario_profit/current_profit - 1)*100:+.1f}%)")
    print(f"   - Vendes perdudes:        ${lost_sales:,.0f}")
    print(f"   - Comandes afectades:     {len(df) - len(scenario_12):,} ({(1 - len(scenario_12)/len(df))*100:.1f}%)")
    
    # Gràfic comparatiu
    fig, ax = plt.subplots(figsize=(10, 6))
    scenarios = ['Situació\nActual', 'Límit\n12%', 'Límit\n15%']
    
    scenario_15 = df[df['order_item_discount_rate'] <= 0.15]
    profits = [
        current_profit,
        scenario_profit,
        scenario_15['benefit_per_order'].sum()
    ]
    
    bars = ax.bar(scenarios, profits, color=['#C9ADA7', '#06A77D', '#4A90E2'], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Afegir valors
    for bar, profit in zip(bars, profits):
        height = bar.get_height()
        diff = profit - current_profit
        diff_pct = (diff / current_profit * 100) if current_profit != 0 else 0
        label = f'${profit/1e6:.1f}M\n({diff_pct:+.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Benefici Total ($)', fontweight='bold')
    ax.set_title('Simulació d\'Escenaris: Impacte de Limitar Descomptes', 
                fontweight='bold', fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_fig('scenario_simulation', 'estudis')
    print("Gràfic guardat: scenario_simulation.png")
    
except Exception as e:
    print(f"Error en Estudi 3: {e}")

