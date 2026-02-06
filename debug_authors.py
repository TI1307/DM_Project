
import pandas as pd
import numpy as np
from app.data_loader import DataLoader

loader = DataLoader()
df = loader.books_df

print("\n--- Books with Unknown Author ---")
unknowns = df[df['Auteur'] == 'Unknown']
print(f"Total books with Unknown author: {len(unknowns)}")

if not unknowns.empty:
    print("\nSample books with Unknown author:")
    print(unknowns[['Titre', 'Titre_clean', 'Source']].head(20))

# Check popular books with Unknown author
popular = loader.borrowings_df['Titre_clean'].value_counts().head(50)
print("\n--- Top Borrowed Books Checking Authors ---")
for title, count in popular.items():
    match = df[df['Titre_clean'] == title]
    if not match.empty:
        author = match.iloc[0]['Auteur']
        if author == 'Unknown':
            print(f"Title: {title} | Count: {count} | Author: {author}")
