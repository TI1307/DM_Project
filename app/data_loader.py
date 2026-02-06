import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import re

class DataLoader:
    """Load and prepare data for the recommendation system"""
    
    def __init__(self, data_dir="data/Clean_Data"):
        """Initialize data loader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.borrowings_df = None
        self.books_df = None  # This will be the full_library_dataset
        self.catalogue_df = None
        self.user_book_matrix = None
        self.user_stats = None
        
        self.load_data()
        self.create_user_book_matrix()
        self.create_user_statistics()
    
    def normalize_text(self, text):
        """Normalize text for matching"""
        text = str(text).strip().upper()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        return text

    def clean_author_name(self, name):
        """Clean author names from encoding issues like ????"""
        if pd.isna(name) or name is None:
            return "Unknown"
        
        name_str = str(name).strip()
        
        # Check for multiple question marks common in encoding errors
        if '???' in name_str or name_str == '?' or name_str == '??':
            return "Unknown"
        
        # Basic cleanup of other common artifacts if any
        name_str = re.sub(r'[\?\[\]]', '', name_str).strip()
        
        if not name_str or name_str.lower() in ['nan', 'none', 'unknown', 'null']:
            return "Unknown"
            
        return name_str
    def load_data(self):
        """Load data files using the same approach as analysis notebook"""
        try:
            # Load borrowings data (look for exact name or variant)
            borrowings_path = self.data_dir / "cleaned_borrowings.xlsx"
            if not borrowings_path.exists():
                # Fallback to general data dir if not in Clean_Data
                borrowings_path = Path("data") / "cleaned_borrowings.xlsx"
            
            self.borrowings_df = pd.read_excel(borrowings_path)
            
            # Clean author names in borrowings immediately
            author_cols = [c for c in self.borrowings_df.columns if 'auteur' in c.lower()]
            for col in author_cols:
                self.borrowings_df[col] = self.borrowings_df[col].apply(self.clean_author_name)
            
            # Ensure required columns exist
            if 'Titre' not in self.borrowings_df.columns:
                raise Exception("Titre column not found in borrowings data")
            
            
            # Load catalogue data
            catalogue_path = self.data_dir / "cleaned_catalogue.xlsx"
            if not catalogue_path.exists():
                catalogue_path = Path("data") / "cleaned_catalogue.xlsx"
            
            self.catalogue_df = pd.read_excel(catalogue_path)
            
            # Clean author names in catalogue
            if 'Auteur' in self.catalogue_df.columns:
                self.catalogue_df['Auteur'] = self.catalogue_df['Auteur'].apply(self.clean_author_name)
            
            
            # Create full library dataset (like in analysis notebook)
            self.create_full_library_dataset()
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def create_full_library_dataset(self):
        """Create full library dataset by merging catalogue with borrowing stats"""
        # Get unique titles from borrowings and catalogue
        borrowing_titles = set(self.borrowings_df['Titre_clean'].unique())
        catalogue_titles = set(self.catalogue_df['Titre_clean'].unique())
        
        # Find matches and missing titles
        matching_titles = borrowing_titles & catalogue_titles
        missing_titles = borrowing_titles - catalogue_titles
        
        # Map borrower categories
        self.borrowings_df['Catégorie_norm'] = self.borrowings_df['Catégorie'].apply(self.normalize_text)
        category_map = {
            '1 ERE ANNEE': '1y', '2 EME ANNEE': '2y', '3 EME ANNEE': '3y',
            '4 EME ANNEE': '4y', '5 ENEE': '5y', 'ENSEIGNANT': 'teacher'
        }
        self.borrowings_df['borrower_group'] = self.borrowings_df['Catégorie_norm'].map(category_map)
        self.borrowings_df = self.borrowings_df[self.borrowings_df['borrower_group'].notna()]
        
        # Process semester
        self.borrowings_df['borrowing duration'] = pd.to_numeric(
            self.borrowings_df['borrowing duration'], errors='coerce'
        )
        median_duration = self.borrowings_df['borrowing duration'].median()
        self.borrowings_df['borrowing duration'] = self.borrowings_df['borrowing duration'].fillna(median_duration)
        self.borrowings_df['Semester'] = self.borrowings_df['borrowing duration'].apply(
            lambda x: 1 if x <= median_duration else 2
        )

        # Prepare a helper to find the most common non-unknown author for any title
        def find_best_author_in_borrowings(titre_clean):
            sample_rows = self.borrowings_df[self.borrowings_df['Titre_clean'] == titre_clean]
            if sample_rows.empty:
                return "Unknown"
            
            # Look across all potential columns
            for col in ['Auteur', 'AUTEUR', 'NOM_AUTEUR']:
                if col in sample_rows.columns:
                    authors = sample_rows[col].dropna()
                    authors = authors[~authors.apply(self.clean_author_name).isin(['Unknown', 'Unknown Author', 'nan', 'None', ''])]
                    if not authors.empty:
                        return authors.value_counts().index[0]
            
            # Last resort: just take the first one even if it's "Unknown" (cleaned)
            first_val = sample_rows.iloc[0].get('Auteur', sample_rows.iloc[0].get('AUTEUR', 'Unknown'))
            return self.clean_author_name(first_val)

        missing_books = []
        for title in missing_titles:
            sample = self.borrowings_df[self.borrowings_df['Titre_clean'] == title].iloc[0]
            author = find_best_author_in_borrowings(title)
            
            missing_books.append({
                'Titre': sample['Titre'],
                'Titre_clean': title,
                'Auteur': author,
                'Cote': sample.get('Cote', f"MISSING_{len(missing_books)}"),
                'ISBN, ISSN...': sample.get('ISBN, ISSN...', np.nan),
                'Source': 'Added from borrowings'
            })
        
        if len(missing_books) > 0:
            missing_df = pd.DataFrame(missing_books)
            self.catalogue_df = pd.concat([self.catalogue_df, missing_df], ignore_index=True)
        
        # Cross-reference: Fill missing authors in catalogue from borrowings
        borrowing_authors = self.borrowings_df.groupby('Titre_clean').apply(
            lambda x: find_best_author_in_borrowings(x.name)
        )
        # Filter out anything that still returned Unknown if we want to preserve catalogue "Unknown"
        borrowing_authors = borrowing_authors[borrowing_authors != 'Unknown']
        
        # Map back to catalogue
        def fill_author(row):
            current = self.clean_author_name(row.get('Auteur', 'Unknown'))
            if current == 'Unknown' and row['Titre_clean'] in borrowing_authors.index:
                return borrowing_authors[row['Titre_clean']]
            return current

        self.catalogue_df['Auteur'] = self.catalogue_df.apply(fill_author, axis=1)
        
        # Aggregate by borrower type
        borrow_agg = self.borrowings_df.groupby(['Titre_clean', 'borrower_group']).size().unstack(fill_value=0)
        borrower_columns = ['1y', '2y', '3y', '4y', '5y', 'teacher']
        for col in borrower_columns:
            if col not in borrow_agg.columns:
                borrow_agg[col] = 0
        borrow_agg = borrow_agg.reset_index()[['Titre_clean'] + borrower_columns]
        
        # Aggregate by semester
        semester_agg = self.borrowings_df.groupby(['Titre_clean', 'Semester']).size().unstack(fill_value=0)
        for sem in [1, 2]:
            if sem not in semester_agg.columns:
                semester_agg[sem] = 0
        semester_agg = semester_agg.rename(columns={1: 'semester_1', 2: 'semester_2'}).reset_index()
        
        # Merge aggregations
        borrowing_features = borrow_agg.merge(semester_agg, on='Titre_clean', how='left')
        
        # Merge with catalogue
        self.books_df = self.catalogue_df.merge(borrowing_features, on='Titre_clean', how='left')
        
        # Fill missing values for borrowing stats
        final_columns = borrower_columns + ['semester_1', 'semester_2']
        self.books_df[final_columns] = self.books_df[final_columns].fillna(0)
        self.books_df['total_borrowed'] = self.books_df[borrower_columns].sum(axis=1)
        
        # 1. COTE-BASED FEATURES (Dewey Decimal Classification)
        self.add_topics_from_cote()
        
        # 2. ISBN-BASED FEATURES
        self.add_isbn_features()
        
        # 3. TEXT-BASED FEATURES
        self.books_df['title_length'] = self.books_df['Titre_clean'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        self.books_df['title_word_count'] = self.books_df['Titre_clean'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        self.books_df['has_author'] = self.books_df['Auteur'].notna().astype(int)

        # 4. COPIES & BORROWING PATTERN
        if 'Nbr. Exp.' in self.books_df.columns:
            self.books_df['num_copies'] = pd.to_numeric(
                self.books_df['Nbr. Exp.'], errors='coerce'
            ).fillna(1)
            self.books_df['borrow_per_copy'] = (
                self.books_df['total_borrowed'] / self.books_df['num_copies']
            )

        # 5. DERIVED Popularity
        threshold = self.books_df['total_borrowed'].quantile(0.75)
        self.books_df['is_popular'] = (
            self.books_df['total_borrowed'] >= threshold
        ).astype(int)
        
        print(f"✓ Created full library dataset with {len(self.books_df)} books")

    def add_topics_from_cote(self):
        """Add topics based on Dewey Decimal Classification from Cote as per analysis notebook"""
        def extract_cote_digits(cote):
            if pd.isna(cote):
                return None
            match = re.search(r'(\d+)', str(cote))
            return match.group(1) if match else None
        
        self.books_df['cote_digits'] = self.books_df['Cote'].apply(extract_cote_digits)
        
        dewey_to_topic_fr = {
            '000': 'Informatique et information', '004': 'Traitement des données, informatique',
            '005': 'Programmation informatique', '006': 'Méthodes informatiques spéciales',
            '100': 'Philosophie et psychologie', '200': 'Religion', '300': 'Sciences sociales',
            '400': 'Langues', '500': 'Sciences naturelles et mathématiques',
            '510': 'Mathématiques', '511': 'Mathématiques générales', '512': 'Algèbre',
            '513': 'Arithmétique', '514': 'Topologie', '515': 'Analyse mathématique',
            '516': 'Géométrie', '517': 'Calcul', '518': 'Analyse numérique',
            '519': 'Probabilités et mathématiques appliquées', '520': 'Astronomie',
            '530': 'Physique', '540': 'Chimie', '550': 'Sciences de la terre',
            '560': 'Paléontologie', '570': 'Sciences de la vie, biologie',
            '580': 'Botanique', '590': 'Zoologie', '600': 'Technologie (sciences appliquées)',
            '610': 'Médecine et santé', '620': 'Ingénierie', '621': 'Physique appliquée',
            '629': 'Autres branches de l\'ingénierie', '630': 'Agriculture',
            '640': 'Économie domestique', '650': 'Gestion et services auxiliaires',
            '660': 'Génie chimique', '670': 'Fabrication',
            '680': 'Fabrication pour usages spécifiques', '690': 'Construction',
            '700': 'Arts et loisirs', '800': 'Littérature', '900': 'Histoire et géographie'
        }
        
        def map_cote_to_topic(cote_digits):
            if pd.isna(cote_digits):
                return 'Non classé'
            cote_str = str(cote_digits)
            if cote_str in dewey_to_topic_fr:
                return dewey_to_topic_fr[cote_str]
            if len(cote_str) >= 2:
                key_2 = cote_str[:2] + '0'
                if key_2 in dewey_to_topic_fr:
                    return dewey_to_topic_fr[key_2]
            if len(cote_str) >= 1:
                key_1 = cote_str[0] + '00'
                if key_1 in dewey_to_topic_fr:
                    return dewey_to_topic_fr[key_1]
            return 'Autre'
        
        self.books_df['topic_fr'] = self.books_df['cote_digits'].apply(map_cote_to_topic)

    def add_isbn_features(self):
        """Clean and extract ISBN features as per analysis notebook"""
        def clean_isbn(isbn_raw):
            if pd.isna(isbn_raw):
                return None
            isbn_str = str(isbn_raw).strip().replace('-', '').replace(' ', '')
            isbn_str = re.sub(r'^(ISBN|isbn|ISSN|issn)[:\s-]*', '', isbn_str)
            isbn_clean = re.sub(r'\D', '', isbn_str)
            if len(isbn_clean) == 10 or len(isbn_clean) == 13:
                return isbn_clean
            if len(isbn_clean) >= 9:
                return isbn_clean
            return None

        def extract_isbn_group_digit(isbn_clean):
            if pd.isna(isbn_clean) or isbn_clean is None:
                return None
            isbn_str = str(isbn_clean)
            if len(isbn_str) == 13 and isbn_str.startswith(('978', '979')):
                return isbn_str[3]
            if len(isbn_str) >= 1:
                return isbn_str[0]
            return None

        self.books_df['isbn_clean'] = self.books_df['ISBN, ISSN...'].apply(clean_isbn)
        self.books_df['isbn_group_digit'] = self.books_df['isbn_clean'].apply(extract_isbn_group_digit)
        
        isbn_country_mapping_fr = {
            '0': 'Pays anglophones', '1': 'Pays anglophones', '2': 'Pays francophones',
            '3': 'Pays germanophones', '4': 'Pays asiatiques', '5': 'Pays slaves/Russie',
            '7': 'Pays scandinaves / autres', '8': 'Autre région', '9': 'Autre région'
        }
        
        self.books_df['isbn_country_fr'] = self.books_df.apply(
            lambda row: isbn_country_mapping_fr.get(str(row['isbn_group_digit']), 'Autre région') 
            if row['isbn_clean'] is not None else 'Sans ISBN', axis=1
        )

    def create_user_book_matrix(self):
        """Create user-book interaction matrix"""
        user_books = self.borrowings_df.groupby(['N° lecteur', 'Titre_clean']).size().reset_index(name='count')
        
        self.user_book_matrix = user_books.pivot_table(
            index='N° lecteur',
            columns='Titre_clean',
            values='count',
            fill_value=0
        )
        
        self.user_book_matrix = (self.user_book_matrix > 0).astype(int)
        print(f"✓ Created user-book matrix: {self.user_book_matrix.shape}")
    
    def create_user_statistics(self):
        """Create comprehensive user statistics"""
        user_borrowing_counts = self.borrowings_df.groupby('N° lecteur').size()
        
        def categorize_user(count):
            if count <= 5:
                return 'Light Reader'
            elif count <= 15:
                return 'Moderate Reader'
            else:
                return 'Heavy Reader'
        
        self.user_stats = pd.DataFrame({
            'user_id': user_borrowing_counts.index,
            'total_borrowings': user_borrowing_counts.values
        })
        
        self.user_stats['category'] = self.user_stats['total_borrowings'].apply(categorize_user)
        
        if 'Catégorie' in self.borrowings_df.columns:
            user_categories = self.borrowings_df.groupby('N° lecteur')['Catégorie'].first()
            self.user_stats = self.user_stats.merge(
                user_categories.rename('user_type'),
                left_on='user_id',
                right_index=True,
                how='left'
            )
        
        print(f"✓ Created user statistics for {len(self.user_stats)} users")
    
    def get_all_books(self):
        """Get all books in library"""
        return self.books_df
    
    def get_borrowed_books_count(self):
        """Get count of unique borrowed books"""
        return self.borrowings_df['Titre_clean'].nunique()
    
    def get_unborrowed_books_count(self):
        """Get count of books never borrowed"""
        return len(self.books_df[self.books_df['total_borrowed'] == 0])
    
    def get_user_borrowed_books(self, user_id):
        """Get list of books borrowed by a user"""
        user_borrowings = self.borrowings_df[self.borrowings_df['N° lecteur'] == user_id]
        return user_borrowings['Titre_clean'].unique().tolist()
    
    def get_user_category(self, user_id):
        """Get user category"""
        if self.user_stats is not None:
            user_data = self.user_stats[self.user_stats['user_id'] == user_id]
            if not user_data.empty:
                return user_data.iloc[0]['category']
        return 'Unknown'
    
    def get_popular_books(self, top_n=10):
        """Get most popular books relying on the enriched books_df metadata"""
        # Count borrowings
        popular = self.borrowings_df['Titre_clean'].value_counts().head(top_n).reset_index()
        popular.columns = ['Titre_clean', 'Count']
        
        result = []
        for _, row in popular.iterrows():
            titre_clean = row['Titre_clean']
            count = row['Count']
            
            # Search in enriched books_df (source of truth for metadata)
            matches = self.books_df[self.books_df['Titre_clean'] == titre_clean]
            
            if not matches.empty:
                info = matches.iloc[0]
                title = info.get('Titre', titre_clean)
                author = info.get('Auteur', 'Unknown')
            else:
                title = titre_clean
                author = 'Unknown'
            
            result.append({
                'Titre': title,
                'Count': count,
                'Auteur': self.clean_author_name(author)
            })
        
        return pd.DataFrame(result)
    
    def get_user_segmentation(self):
        """Get user segmentation statistics"""
        if self.user_stats is not None:
            result = self.user_stats['category'].value_counts().reset_index()
            result.columns = ['category', 'count']
            return result
        return pd.DataFrame()
    
    def get_borrowing_by_user_type(self):
        """Get borrowing statistics by user type"""
        if 'Catégorie' in self.borrowings_df.columns:
            result = self.borrowings_df['Catégorie'].value_counts().reset_index()
            result.columns = ['category', 'count']
            return result
        return pd.DataFrame()
    
    def get_statistics(self):
        """Get dataset statistics"""
        total_books = len(self.books_df)
        borrowed_books = self.get_borrowed_books_count()
        unborrowed_books = self.get_unborrowed_books_count()
        unique_users = self.borrowings_df['N° lecteur'].nunique()
        total_borrowings = len(self.borrowings_df)
        
        stats = {
            'total_books': total_books,
            'borrowed_books': borrowed_books,
            'unborrowed_books': unborrowed_books,
            'total_borrowings': total_borrowings,
            'unique_users': unique_users,
            'avg_borrowings_per_user': total_borrowings / unique_users if unique_users > 0 else 0,
            'avg_borrowings_per_book': total_borrowings / borrowed_books if borrowed_books > 0 else 0
        }
        
        if self.user_stats is not None:
            segmentation = self.user_stats['category'].value_counts()
            stats['light_readers'] = segmentation.get('Light Reader', 0)
            stats['moderate_readers'] = segmentation.get('Moderate Reader', 0)
            stats['heavy_readers'] = segmentation.get('Heavy Reader', 0)
        
        return stats
    
    def get_category_distribution(self):
        """Get distribution of books by category"""
        if 'topic_fr' in self.books_df.columns:
            dist = self.books_df['topic_fr'].value_counts()
            # Remove 'Non classé' and 'Autre' if you want
            dist = dist[~dist.index.isin(['Non classé', 'Autre'])]
            return dist
        return pd.Series()
    
    def get_author_statistics(self, top_n=10):
        """Get top authors by number of books with strict filtering"""
        if 'Auteur' in self.books_df.columns:
            # Clean first
            self.books_df['Auteur'] = self.books_df['Auteur'].apply(self.clean_author_name)
            
            author_counts = self.books_df['Auteur'].value_counts()
            
            # Remove all variants of Unknown
            unknown_variants = ['Unknown', 'Unknown Author', 'nan', 'None', '', ' ']
            author_counts = author_counts[~author_counts.index.isin(unknown_variants)]
            
            # Further filter out single characters or question mark strings if any escaped
            author_counts = author_counts[author_counts.index.str.len() > 2]
            
            author_counts = author_counts.head(top_n).reset_index()
            author_counts.columns = ['Author', 'Book Count']
            return author_counts
        return pd.DataFrame()