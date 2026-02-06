import pandas as pd
from pathlib import Path

def inspect():
    data_dir = Path("data/Clean_Data")
    borrowings_path = data_dir / "cleaned_borrowings.xlsx"
    catalogue_path = data_dir / "cleaned_catalogue.xlsx"
    
    if not borrowings_path.exists():
        borrowings_path = Path("data/cleaned_borrowings.xlsx")
    if not catalogue_path.exists():
        catalogue_path = Path("data/cleaned_catalogue.xlsx")
        
    print(f"Loading {borrowings_path}")
    df_b = pd.read_excel(borrowings_path)
    print("Borrowings columns:", df_b.columns.tolist())
    
    author_cols_b = [c for c in df_b.columns if 'auteur' in c.lower()]
    print("Author-like columns in borrowings:", author_cols_b)
    for col in author_cols_b:
        print(f"Sample from {col}:", df_b[col].dropna().unique()[:5])
        
    print(f"\nLoading {catalogue_path}")
    df_c = pd.read_excel(catalogue_path)
    print("Catalogue columns:", df_c.columns.tolist())
    author_cols_c = [c for c in df_c.columns if 'auteur' in c.lower()]
    print("Author-like columns in catalogue:", author_cols_c)
    for col in author_cols_c:
        print(f"Sample from {col}:", df_c[col].dropna().unique()[:5])

if __name__ == "__main__":
    inspect()
