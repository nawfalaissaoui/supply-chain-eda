# %%
import pandas as pd

# %%
def detectar_candidats_a_eliminar(csv_path):
    print(f"INICIANT AUDITORIA DE DADES: {csv_path}")
    print("=" * 60)
    
    # Càrrega de dades
    df = pd.read_csv(csv_path)
        
    print(f"Dimensions: {df.shape[0]} files x {df.shape[1]} columnes")
    
    # Llistes per emmagatzemar troballes
    propostes_eliminar = set()
    
    # ---------------------------------------------------------
    # 1 Columnes amb molts NULS (Qualitat de dada)
    # ---------------------------------------------------------
    print("\n[1] Columnes amb excessius valors nuls (>85%):")
    null_percent = df.isnull().mean()
    high_nulls = null_percent[null_percent > 0.85].index.tolist()
    if high_nulls:
        for col in high_nulls:
            print(f"   - {col} ({null_percent[col]*100:.1f}% nuls)")
            propostes_eliminar.add(col)
    else:
        print("   Cap columna supera el llindar de nuls.")

    # ---------------------------------------------------------
    # 2 Columnes CONSTANTS (Informació zero)
    # ---------------------------------------------------------
    print("\n[2] Columnes Constants (un sol valor):")
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        for col in const_cols:
            unique_val = df[col].unique()
            print(f"   - {col} (Valor únic: {unique_val})")
            propostes_eliminar.add(col)
    else:
        print("   No s'han trobat columnes constants.")

    # ---------------------------------------------------------
    # 3 Columnes DUPLICADES EXACTES (Redundància tècnica)
    # ---------------------------------------------------------
    print("\n[3] Columnes Duplicades Exactes (valors idèntics):")
    
    dup_pairs = []
    
    # Comparem columnes del mateix tipus
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            if df[col1].dtype == df[col2].dtype:
                if df[col1].equals(df[col2]):
                    dup_pairs.append((col1, col2))
                    propostes_eliminar.add(col2)
    
    if dup_pairs:
        for c1, c2 in dup_pairs:
            print(f"   - {c2} és idèntica a {c1}")
    else:
        print("   No s'han trobat columnes idèntiques.")

    # ---------------------------------------------------------
    # 4 Dades PERSONALS i URLs (Privacitat i Soroll)
    # ---------------------------------------------------------
    print("\n[4] Dades Personals o Irrellevants (per nom):")
    keywords = ['image', 'url', 'password', 'email', 'fname', 'lname', 'lastname', 'firstname']
    sensitive_cols = [col for col in df.columns if any(k in col.lower() for k in keywords)]
    
    if sensitive_cols:
        for col in sensitive_cols:
            print(f"   - {col}")
            propostes_eliminar.add(col)
    else:
        print("   No s'han detectat dades sensibles.")

    # ---------------------------------------------------------
    # 5 IDs Tècnics i Relacions 1 a 1 (Redundància Semàntica)
    # ---------------------------------------------------------
    print("\n[5] Relacions 1-a-1 (ID vs Nom):")
    
    candidates_id = [c for c in df.columns if '_id' in c]
    candidates_name = [c for c in df.columns if '_name' in c or 'city' in c or 'state' in c]
    
    for id_col in candidates_id:
        root = id_col.replace('_id', '')
        matches = [c for c in candidates_name if root in c]
        
        for name_col in matches:
            try:
                relation = df.groupby(id_col)[name_col].nunique().max()
                if relation == 1:
                    print(f"   - {id_col} <--> {name_col} (Redundants)")
                    propostes_eliminar.add(id_col)
            except:
                pass

    # ---------------------------------------------------------
    # RESUM FINAL
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("PROPOSTA FINAL D'ELIMINACIÓ")
    print("="*60)
    print("-" * 30)
    print("cols_to_drop = [")
    for col in sorted(list(propostes_eliminar)):
        print(f"    '{col}',")
    print("]")
    print("-" * 30)

# %%
# Executar la funció principal
if __name__ == "__main__":
    detectar_candidats_a_eliminar('./data/cadena_subministrament_2015_2018.csv')