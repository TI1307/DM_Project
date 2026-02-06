import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Load and prepare data for the recommendation system"""
    
    def __init__(self, data_dir="data"):
        """Initialize data loader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.borrowings_df = None
        self.books_df = None  # This will contain ALL books (borrowed and unborrowed)
        self.clustering_df = None
        self.user_book_matrix = None
        self.user_stats = None
        
        self.load_data()
        self.prepare_data()
        self.create_user_statistics()
    
    def load_data(self):
        """Load data files"""
        try:
            # Load borrowings data
            borrowings_path = self.data_dir / "cleaned_borrowings.xlsx"
            self.borrowings_df = pd.read_excel(borrowings_path)
            print(f"✓ Loaded {len(self.borrowings_df)} borrowing records")
            
            # Load FULL library catalog (including unborrowed books)
            full_library_path = self.data_dir / "full_library_dataset.csv"
            if full_library_path.exists():
                self.books_df = pd.read_csv(full_library_path, encoding='utf-8')
                print(f"✓ Loaded {len(self.books_df)} books from full library catalog")
            else:
                # Fallback to old dataset
                books_path = self.data_dir / "unified_library_with_topics_isbn.csv"
                self.books_df = pd.read_csv(books_path, encoding='utf-8')
                print(f"✓ Loaded {len(self.books_df)} book records")
            
            # Load clustering data if available
            clustering_path = self.data_dir / "Final_data_for_clustering.xlsx"
            if clustering_path.exists():
                self.clustering_df = pd.read_excel(clustering_path)
                print(f"✓ Loaded clustering dataset with {len(self.clustering_df)} records")
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def prepare_data(self):
        """Prepare and clean data"""
        # Clean book titles in borrowings
        if 'Titre' in self.borrowings_df.columns:
            self.borrowings_df['Titre'] = self.borrowings_df['Titre'].fillna('Unknown')
        
        # Handle Titre_clean column if it exists
        if 'Titre_clean' not in self.borrowings_df.columns and 'Titre' in self.borrowings_df.columns:
            self.borrowings_df['Titre_clean'] = self.borrowings_df['Titre']
        
        # Clean book titles in full library dataset
        if 'Titre' in self.books_df.columns:
            self.books_df['Titre'] = self.books_df['Titre'].fillna('Unknown')
        
        if 'Titre_clean' not in self.books_df.columns and 'Titre' in self.books_df.columns:
            self.books_df['Titre_clean'] = self.books_df['Titre']
        
        # Create user-book matrix for recommendations
        self.create_user_book_matrix()
    
    def create_user_book_matrix(self):
        """Create user-book interaction matrix"""
        # Determine which title column to use
        title_col = 'Titre_clean' if 'Titre_clean' in self.borrowings_df.columns else 'Titre'
        
        # Group by user and book
        user_books = self.borrowings_df.groupby(['N° lecteur', title_col]).size().reset_index(name='count')
        
        # Create pivot table
        self.user_book_matrix = user_books.pivot_table(
            index='N° lecteur',
            columns=title_col,
            values='count',
            fill_value=0
        )
        
        # Convert to binary (borrowed or not)
        self.user_book_matrix = (self.user_book_matrix > 0).astype(int)
        
        print(f"✓ Created user-book matrix: {self.user_book_matrix.shape}")
    
    def create_user_statistics(self):
        """Create comprehensive user statistics"""
        # Calculate borrowing frequency per user
        user_borrowing_counts = self.borrowings_df.groupby('N° lecteur').size()
        
        # Categorize users by borrowing frequency
        def categorize_user(count):
            if count <= 5:
                return 'Light Reader'
            elif count <= 15:
                return 'Moderate Reader'
            else:
                return 'Heavy Reader'
        
        # Create user statistics dataframe
        self.user_stats = pd.DataFrame({
            'user_id': user_borrowing_counts.index,
            'total_borrowings': user_borrowing_counts.values
        })
        
        self.user_stats['category'] = self.user_stats['total_borrowings'].apply(categorize_user)
        
        # Add user category from original data if available
        if 'Catégorie' in self.borrowings_df.columns:
            user_categories = self.borrowings_df.groupby('N° lecteur')['Catégorie'].first()
            self.user_stats = self.user_stats.merge(
                user_categories.rename('user_type'),
                left_on='user_id',
                right_index=True,
                how='left'
            )
        
        print(f"✓ Created user statistics for {len(self.user_stats)} users")
    
    def get_book_info(self, title):
        """Get detailed information about a book from full library
        
        Args:
            title: Book title
            
        Returns:
            DataFrame with book information
        """
        # Try with Titre_clean first, then Titre
        result = self.books_df[self.books_df['Titre_clean'] == title] if 'Titre_clean' in self.books_df.columns else pd.DataFrame()
        if result.empty:
            result = self.books_df[self.books_df['Titre'] == title]
        return result
    
    def get_all_books(self):
        """Get all books in library (borrowed and unborrowed)
        
        Returns:
            DataFrame with all books
        """
        return self.books_df
    
    def get_borrowed_books_count(self):
        """Get count of books that have been borrowed at least once
        
        Returns:
            int: Number of unique borrowed books
        """
        title_col = 'Titre_clean' if 'Titre_clean' in self.borrowings_df.columns else 'Titre'
        return self.borrowings_df[title_col].nunique()
    
    def get_unborrowed_books_count(self):
        """Get count of books that have never been borrowed
        
        Returns:
            int: Number of unborrowed books
        """
        if 'total_borrowed' in self.books_df.columns:
            return len(self.books_df[self.books_df['total_borrowed'] == 0])
        return 0
    
    def get_user_borrowed_books(self, user_id):
        """Get list of books borrowed by a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of book titles
        """
        user_borrowings = self.borrowings_df[self.borrowings_df['N° lecteur'] == user_id]
        title_col = 'Titre_clean' if 'Titre_clean' in self.borrowings_df.columns else 'Titre'
        return user_borrowings[title_col].unique().tolist()
    
    def get_user_category(self, user_id):
        """Get user category (Light/Moderate/Heavy Reader)
        
        Args:
            user_id: User ID
            
        Returns:
            User category string
        """
        if self.user_stats is not None:
            user_data = self.user_stats[self.user_stats['user_id'] == user_id]
            if not user_data.empty:
                return user_data.iloc[0]['category']
        return 'Unknown'
    
    def get_popular_books(self, top_n=10):
        """Get most popular books
        
        Args:
            top_n: Number of top books to return
            
        Returns:
            DataFrame with popular books
        """
        title_col = 'Titre_clean' if 'Titre_clean' in self.borrowings_df.columns else 'Titre'
        popular = self.borrowings_df[title_col].value_counts().head(top_n).reset_index()
        popular.columns = ['Titre', 'Count']
        
        # Add book details from full library
        popular = popular.merge(
            self.books_df[['Titre', 'Auteur']].drop_duplicates('Titre'),
            left_on='Titre',
            right_on='Titre',
            how='left'
        )
        
        # Add topic if available
        if 'topic_fr' in self.books_df.columns:
            topic_data = self.books_df[['Titre', 'topic_fr']].drop_duplicates('Titre')
            popular = popular.merge(topic_data, on='Titre', how='left')
        
        return popular
    
    def get_books_by_category(self, category):
        """Get books in a specific category
        
        Args:
            category: Category name
            
        Returns:
            DataFrame with books in category
        """
        if 'topic_fr' not in self.books_df.columns:
            return pd.DataFrame()
        
        return self.books_df[self.books_df['topic_fr'] == category]
    
    def get_user_segmentation(self):
        """Get user segmentation statistics
        
        Returns:
            DataFrame with user segmentation counts
        """
        if self.user_stats is not None:
            return self.user_stats['category'].value_counts().reset_index()
        return pd.DataFrame()
    
    def get_borrowing_by_user_type(self):
        """Get borrowing statistics by user type (student year, teacher, etc.)
        
        Returns:
            DataFrame with borrowing counts by user type
        """
        if 'Catégorie' in self.borrowings_df.columns:
            return self.borrowings_df['Catégorie'].value_counts().reset_index()
        return pd.DataFrame()
    
    def get_statistics(self):
        """Get dataset statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_books': len(self.books_df),
            'borrowed_books': self.get_borrowed_books_count(),
            'unborrowed_books': self.get_unborrowed_books_count(),
            'total_borrowings': len(self.borrowings_df),
            'unique_users': self.borrowings_df['N° lecteur'].nunique(),
            'avg_borrowings_per_user': len(self.borrowings_df) / self.borrowings_df['N° lecteur'].nunique(),
            'avg_borrowings_per_book': len(self.borrowings_df) / self.get_borrowed_books_count() if self.get_borrowed_books_count() > 0 else 0
        }
        
        # Add user segmentation stats
        if self.user_stats is not None:
            segmentation = self.user_stats['category'].value_counts()
            stats['light_readers'] = segmentation.get('Light Reader', 0)
            stats['moderate_readers'] = segmentation.get('Moderate Reader', 0)
            stats['heavy_readers'] = segmentation.get('Heavy Reader', 0)
        
        return stats
    
    def get_category_distribution(self):
        """Get distribution of books by category from full library
        
        Returns:
            Series with category counts
        """
        if 'topic_fr' in self.books_df.columns:
            return self.books_df['topic_fr'].value_counts()
        return pd.Series()
    
    def get_author_statistics(self, top_n=10):
        """Get top authors by number of books in full library
        
        Args:
            top_n: Number of top authors to return
            
        Returns:
            DataFrame with author statistics
        """
        if 'Auteur' in self.books_df.columns:
            author_counts = self.books_df['Auteur'].value_counts().head(top_n).reset_index()
            author_counts.columns = ['Author', 'Book Count']
            return author_counts
        return pd.DataFrame()
    
    def get_clustering_data(self):
        """Get clustering dataset if available
        
        Returns:
            DataFrame with clustering data or None
        """
        return self.clustering_df