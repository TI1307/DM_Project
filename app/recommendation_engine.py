import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle
from pathlib import Path

class RecommendationEngine:
    """Book recommendation engine using Association Rules"""
    
    def __init__(self, data_loader, min_support=0.005, min_confidence=0.3):
        """Initialize recommendation engine
        
        Args:
            data_loader: DataLoader instance
            min_support: Minimum support threshold for FP-Growth
            min_confidence: Minimum confidence for association rules
        """
        self.data_loader = data_loader
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        self.frequent_itemsets = None
        self.association_rules_df = None
        self.transaction_df = None
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Try to load pre-trained model, otherwise train new one
        if not self.load_model():
            self.train_model()
    
    def prepare_transactions(self):
        """Prepare transaction data for FP-Growth
        
        Returns:
            DataFrame in transaction format
        """
        # Determine which title column to use
        title_col = 'Titre_clean' if 'Titre_clean' in self.data_loader.borrowings_df.columns else 'Titre'
        
        # Group books by user
        transactions = self.data_loader.borrowings_df.groupby('N° lecteur')[title_col].apply(list).values
        
        # Convert to transaction format
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        return df
    
    def train_model(self):
        """Train FP-Growth and generate association rules"""
        print("📄 Training recommendation model...")
        
        # Prepare transactions
        self.transaction_df = self.prepare_transactions()
        print(f"✓ Prepared {len(self.transaction_df)} transactions with {len(self.transaction_df.columns)} unique books")
        
        # Apply FP-Growth algorithm
        print(f"⚙️  Running FP-Growth with min_support={self.min_support}...")
        self.frequent_itemsets = fpgrowth(
            self.transaction_df, 
            min_support=self.min_support, 
            use_colnames=True
        )
        print(f"✓ Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            print("⚙️  Generating association rules...")
            self.association_rules_df = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
            
            # Add additional metrics
            if len(self.association_rules_df) > 0:
                # Calculate conviction (measure of implication strength)
                self.association_rules_df['conviction'] = (
                    (1 - self.association_rules_df['consequent support']) / 
                    (1 - self.association_rules_df['confidence'])
                )
                
                # Replace infinite values with a large number
                self.association_rules_df['conviction'] = self.association_rules_df['conviction'].replace(
                    [np.inf, -np.inf], 999
                )
                
                self.association_rules_df = self.association_rules_df.sort_values(
                    'confidence', 
                    ascending=False
                )
                print(f"✓ Generated {len(self.association_rules_df)} association rules")
                print(f"  - Average Confidence: {self.association_rules_df['confidence'].mean():.3f}")
                print(f"  - Average Lift: {self.association_rules_df['lift'].mean():.3f}")
                print(f"  - Strong Rules (Lift>1.5): {len(self.association_rules_df[self.association_rules_df['lift']>1.5])}")
            else:
                print("⚠️  No association rules found with current thresholds")
        else:
            print("⚠️  No frequent itemsets found")
        
        # Save model
        self.save_model()
    
    def get_recommendations(self, book_title, num_recommendations=10):
        """Get book recommendations based on a given book
        
        Args:
            book_title: Title of the book
            num_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return None
        
        # Find rules where the book is in antecedents
        recommendations = []
        
        for idx, rule in self.association_rules_df.iterrows():
            antecedents = rule['antecedents']
            consequents = rule['consequents']
            
            # Check if book is in antecedents
            if book_title in antecedents:
                for book in consequents:
                    if book != book_title:
                        recommendations.append({
                            'Titre': book,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support'],
                            'conviction': rule.get('conviction', 1.0)
                        })
        
        if not recommendations:
            # If no direct rules, try finding rules where book is in consequents
            for idx, rule in self.association_rules_df.iterrows():
                consequents = rule['consequents']
                antecedents = rule['antecedents']
                
                if book_title in consequents:
                    for book in antecedents:
                        if book != book_title:
                            recommendations.append({
                                'Titre': book,
                                'confidence': rule['confidence'] * 0.8,  # Lower confidence
                                'lift': rule['lift'],
                                'support': rule['support'],
                                'conviction': rule.get('conviction', 1.0) * 0.8
                            })
        
        if not recommendations:
            # Fallback to popular books
            return self.get_popular_recommendations(num_recommendations)
        
        # Convert to DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Remove duplicates and sort
        recommendations_df = recommendations_df.drop_duplicates('Titre')
        recommendations_df = recommendations_df.sort_values('confidence', ascending=False)
        
        # Limit to requested number
        recommendations_df = recommendations_df.head(num_recommendations)
        
        # Add book details
        recommendations_df = recommendations_df.merge(
            self.data_loader.books_df[['Titre', 'Auteur_merged1', 'topic_fr', 'Editeur']].drop_duplicates('Titre'),
            on='Titre',
            how='left'
        )
        
        return recommendations_df
    
    def get_popular_recommendations(self, num_recommendations=10):
        """Get popular books as fallback recommendations
        
        Args:
            num_recommendations: Number of recommendations
            
        Returns:
            DataFrame with popular books
        """
        popular = self.data_loader.get_popular_books(num_recommendations)
        popular['confidence'] = 1.0
        popular['lift'] = 1.0
        popular['support'] = 0.0
        return popular
    
    def get_recommendations_for_user(self, user_id, num_recommendations=10):
        """Get personalized recommendations for a user
        
        Args:
            user_id: User ID
            num_recommendations: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        # Get user's borrowed books
        user_books = self.data_loader.get_user_borrowed_books(user_id)
        
        if not user_books:
            return self.get_popular_recommendations(num_recommendations)
        
        # Get recommendations for each book
        all_recommendations = []
        
        for book in user_books[:10]:  # Limit to last 10 books to avoid too many queries
            recs = self.get_recommendations(book, num_recommendations=5)
            if recs is not None and len(recs) > 0:
                all_recommendations.append(recs)
        
        if not all_recommendations:
            return self.get_popular_recommendations(num_recommendations)
        
        # Combine and aggregate
        combined = pd.concat(all_recommendations, ignore_index=True)
        
        # Remove books user already borrowed
        combined = combined[~combined['Titre'].isin(user_books)]
        
        # Aggregate by book (average confidence)
        agg_dict = {'confidence': 'mean'}
        if 'Auteur_merged1' in combined.columns:
            agg_dict['Auteur_merged1'] = 'first'
        if 'topic_fr' in combined.columns:
            agg_dict['topic_fr'] = 'first'
        if 'lift' in combined.columns:
            agg_dict['lift'] = 'mean'
        
        aggregated = combined.groupby('Titre').agg(agg_dict).reset_index()
        
        # Sort and limit
        aggregated = aggregated.sort_values('confidence', ascending=False).head(num_recommendations)
        
        return aggregated
    
    def get_rule_statistics(self):
        """Get statistics about association rules
        
        Returns:
            Dictionary with rule statistics
        """
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return {
                'total_rules': 0,
                'avg_confidence': 0,
                'avg_lift': 0,
                'avg_support': 0,
                'strong_rules': 0
            }
        
        stats = {
            'total_rules': len(self.association_rules_df),
            'avg_confidence': self.association_rules_df['confidence'].mean(),
            'avg_lift': self.association_rules_df['lift'].mean(),
            'avg_support': self.association_rules_df['support'].mean(),
            'strong_rules': len(self.association_rules_df[self.association_rules_df['lift'] > 1.5]),
            'max_confidence': self.association_rules_df['confidence'].max(),
            'max_lift': self.association_rules_df['lift'].max(),
            'total_itemsets': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0
        }
        
        return stats
    
    def get_top_rules(self, top_n=20, sort_by='confidence'):
        """Get top association rules
        
        Args:
            top_n: Number of top rules to return
            sort_by: Metric to sort by ('confidence', 'lift', 'support')
            
        Returns:
            DataFrame with top rules
        """
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return pd.DataFrame()
        
        # Convert frozensets to strings for display
        rules_display = self.association_rules_df.copy()
        rules_display['antecedents_str'] = rules_display['antecedents'].apply(
            lambda x: ', '.join(list(x))
        )
        rules_display['consequents_str'] = rules_display['consequents'].apply(
            lambda x: ', '.join(list(x))
        )
        
        # Sort and select top rules
        top_rules = rules_display.sort_values(sort_by, ascending=False).head(top_n)
        
        return top_rules[['antecedents_str', 'consequents_str', 'support', 
                         'confidence', 'lift', 'conviction']]
    
    def analyze_book_associations(self, book_title):
        """Analyze all associations for a specific book
        
        Args:
            book_title: Title of the book
            
        Returns:
            Dictionary with association analysis
        """
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return None
        
        # Rules where book is antecedent
        antecedent_rules = []
        for idx, rule in self.association_rules_df.iterrows():
            if book_title in rule['antecedents']:
                antecedent_rules.append(rule)
        
        # Rules where book is consequent
        consequent_rules = []
        for idx, rule in self.association_rules_df.iterrows():
            if book_title in rule['consequents']:
                consequent_rules.append(rule)
        
        analysis = {
            'book_title': book_title,
            'as_antecedent': len(antecedent_rules),
            'as_consequent': len(consequent_rules),
            'total_associations': len(antecedent_rules) + len(consequent_rules),
            'antecedent_rules': pd.DataFrame(antecedent_rules) if antecedent_rules else pd.DataFrame(),
            'consequent_rules': pd.DataFrame(consequent_rules) if consequent_rules else pd.DataFrame()
        }
        
        return analysis
    
    def save_model(self):
        """Save trained model to disk"""
        model_data = {
            'frequent_itemsets': self.frequent_itemsets,
            'association_rules': self.association_rules_df,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'transaction_df': self.transaction_df
        }
        
        model_file = self.model_path / "recommendation_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {model_file}")
    
    def load_model(self):
        """Load pre-trained model from disk
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_file = self.model_path / "recommendation_model.pkl"
        
        if not model_file.exists():
            return False
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.frequent_itemsets = model_data['frequent_itemsets']
            self.association_rules_df = model_data['association_rules']
            self.min_support = model_data['min_support']
            self.min_confidence = model_data['min_confidence']
            self.transaction_df = model_data.get('transaction_df', None)
            
            print(f"✓ Model loaded from {model_file}")
            print(f"  - {len(self.frequent_itemsets)} frequent itemsets")
            print(f"  - {len(self.association_rules_df)} association rules")
            
            return True
        except Exception as e:
            print(f"⚠️  Error loading model: {str(e)}")
            return False
    
    def retrain_model(self, min_support=None, min_confidence=None):
        """Retrain model with new parameters
        
        Args:
            min_support: New minimum support threshold
            min_confidence: New minimum confidence threshold
        """
        if min_support is not None:
            self.min_support = min_support
        if min_confidence is not None:
            self.min_confidence = min_confidence
        
        print(f"🔄 Retraining model with support={self.min_support}, confidence={self.min_confidence}")
        self.train_model()