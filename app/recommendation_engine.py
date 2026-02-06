import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

import pickle
from pathlib import Path


class RecommendationEngine:
    """Book recommendation engine using Association Rules"""

    def __init__(self, data_loader, min_support=0.005, min_confidence=0.3):
        """
        Initialize recommendation engine

        Args:
            data_loader: DataLoader instance
            min_support: Minimum support threshold
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
        """Prepare transaction data for FP-Growth"""
        df = self.data_loader.borrowings_df.copy()

        # Use Titre_clean column
        borrowings_transactions = (
            df.groupby("N° lecteur")["Titre_clean"]
            .apply(list)
            .reset_index()
        )

        transactions = borrowings_transactions["Titre_clean"].tolist()

        # Transform to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        borrowing_df = pd.DataFrame(te_ary, columns=te.columns_)

        return borrowing_df

    def train_model(self):
        """Train FP-Growth and generate association rules"""
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            self.association_rules_df = association_rules(
                self.frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence,
            )

            if len(self.association_rules_df) > 0:
                # Add conviction metric
                self.association_rules_df["conviction"] = (
                    (1 - self.association_rules_df["consequent support"])
                    / (1 - self.association_rules_df["confidence"])
                )
                self.association_rules_df["conviction"] = (
                    self.association_rules_df["conviction"].replace([np.inf, -np.inf], 999)
                )

                # Sort by lift for better diversity
                self.association_rules_df = self.association_rules_df.sort_values(
                    ["lift", "confidence"],
                    ascending=False,
                )

        # Save model
        self.save_model()

    def get_recommendations(self, book_title, num_recommendations=10):
        """Get book recommendations with better diversity"""
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return None

        # Convert to Titre_clean for matching
        book_title_clean = book_title.strip().upper()
        
        recommendations = []
        seen_books = set([book_title_clean])

        # Get rules where book is in antecedents (sorted by lift for diversity)
        for _, rule in self.association_rules_df.iterrows():
            antecedents = rule["antecedents"]
            consequents = rule["consequents"]

            if book_title_clean in antecedents:
                for book in consequents:
                    if book not in seen_books:
                        recommendations.append({
                            "Titre_clean": book,
                            "score": rule["lift"] * rule["confidence"],  # Combined score
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                        })
                        seen_books.add(book)

        # If not enough, try where book is in consequents
        if len(recommendations) < num_recommendations:
            for _, rule in self.association_rules_df.iterrows():
                consequents = rule["consequents"]
                antecedents = rule["antecedents"]

                if book_title_clean in consequents:
                    for book in antecedents:
                        if book not in seen_books:
                            recommendations.append({
                                "Titre_clean": book,
                                "score": rule["lift"] * rule["confidence"] * 0.8,
                                "confidence": rule["confidence"] * 0.8,
                                "lift": rule["lift"],
                            })
                            seen_books.add(book)

        # Fallback to popular books
        if not recommendations:
            return self.get_popular_recommendations(num_recommendations)

        # Convert to DataFrame and sort by combined score for diversity
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.drop_duplicates("Titre_clean")
        recommendations_df = recommendations_df.sort_values("score", ascending=False)
        recommendations_df = recommendations_df.head(num_recommendations)

        # Add book details from books_df (merged dataset with better metadata)
        for idx, row in recommendations_df.iterrows():
            book_data = self.data_loader.books_df[
                self.data_loader.books_df['Titre_clean'] == row['Titre_clean']
            ]
            
            if not book_data.empty:
                book_data = book_data.iloc[0]
                recommendations_df.at[idx, 'Titre'] = book_data.get('Titre', row['Titre_clean'])
                author = book_data.get('Auteur', 'Unknown')
                recommendations_df.at[idx, 'Auteur'] = self.data_loader.clean_author_name(author)
            else:
                # Fallback to borrowings if not in books_df
                fallback_data = self.data_loader.borrowings_df[
                    self.data_loader.borrowings_df['Titre_clean'] == row['Titre_clean']
                ]
                if not fallback_data.empty:
                    fallback_data = fallback_data.iloc[0]
                    recommendations_df.at[idx, 'Titre'] = fallback_data.get('Titre', row['Titre_clean'])
                    author = fallback_data.get('Auteur', fallback_data.get('AUTEUR', fallback_data.get('NOM_AUTEUR', 'Unknown')))
                    recommendations_df.at[idx, 'Auteur'] = self.data_loader.clean_author_name(author)

        # Reorder columns
        final_cols = ['Titre', 'Auteur']
        recommendations_df = recommendations_df[final_cols]

        return recommendations_df

    def get_popular_recommendations(self, num_recommendations=10):
        """Get popular books as fallback"""
        popular = self.data_loader.get_popular_books(num_recommendations)
        return popular

    def get_recommendations_for_user(self, user_id, num_recommendations=10, method="association"):
        """Get personalized recommendations for a user"""
        if method == "clustering_category":
            return self.get_recommendations_by_cluster(user_id, num_recommendations, mode="category")
        elif method == "clustering_advanced":
            return self.get_recommendations_by_cluster(user_id, num_recommendations, mode="advanced")

        # Association rules logic
        user_books = self.data_loader.get_user_borrowed_books(user_id)

        if not user_books:
            return self.get_popular_recommendations(num_recommendations)

        all_recommendations = []
        seen_books = set(user_books)

        # Get recommendations from user's borrowed books
        for book in user_books[:15]:
            recs = self.get_recommendations(book, num_recommendations=15)
            if recs is not None and len(recs) > 0:
                # Filter out already seen books
                recs = recs[~recs['Titre'].isin(seen_books)]
                all_recommendations.append(recs)

        if not all_recommendations:
            return self.get_recommendations_by_cluster(user_id, num_recommendations, mode="category")

        # Combine all recommendations
        combined = pd.concat(all_recommendations, ignore_index=True)

        # Remove duplicates and take top N
        combined = combined.drop_duplicates('Titre').head(num_recommendations)

        return combined

    def get_recommendations_by_clustering(self, user_id, clustering_analysis, num_recommendations=3):
        """Get recommendations based on clustering profile"""
        # Ensure clustering results exist
        if clustering_analysis.df_profiles is None or 'kmeans_cluster' not in clustering_analysis.df_profiles.columns:
            return self.get_popular_recommendations(num_recommendations)
            
        # Find user's cluster
        user_profile = clustering_analysis.df_profiles[clustering_analysis.df_profiles['N° lecteur'] == user_id]
        if user_profile.empty:
            return self.get_popular_recommendations(num_recommendations)
            
        cluster_id = user_profile.iloc[0]['kmeans_cluster']
        
        # Find other users in the same cluster
        similar_users = clustering_analysis.df_profiles[
            clustering_analysis.df_profiles['kmeans_cluster'] == cluster_id
        ]['N° lecteur'].tolist()
        
        # Find books borrowed by these similar users
        cluster_borrowings = self.data_loader.borrowings_df[
            self.data_loader.borrowings_df['N° lecteur'].isin(similar_users)
        ]
        
        # Get user's own borrowed books to exclude them
        user_books = self.data_loader.get_user_borrowed_books(user_id)
        
        # Rank books by popularity within the cluster
        cluster_popular = cluster_borrowings[
            ~cluster_borrowings['Titre_clean'].isin(user_books)
        ]['Titre_clean'].value_counts().head(num_recommendations * 2).reset_index()
        cluster_popular.columns = ['Titre_clean', 'count']
        
        if cluster_popular.empty:
            return self.get_popular_recommendations(num_recommendations)
            
        # Add metadata and format
        result = []
        for _, row in cluster_popular.head(num_recommendations).iterrows():
            titre_clean = row['Titre_clean']
            
            # Get metadata from books_df
            book_data = self.data_loader.books_df[self.data_loader.books_df['Titre_clean'] == titre_clean]
            if not book_data.empty:
                book_info = book_data.iloc[0]
                title = book_info.get('Titre', titre_clean)
                author = book_info.get('Auteur', 'Unknown')
            else:
                title = titre_clean
                author = 'Unknown'
                
            result.append({
                'Titre': title,
                'Auteur': self.data_loader.clean_author_name(author)
            })
            
        return pd.DataFrame(result)

    def get_book_recommendations_by_clustering(self, book_title, clustering_analysis, num_recommendations=3):
        """Get hybrid recommendations blending specific associations with behavioral context"""
        
        # 1. Get Specific Recommendations (Association Rules)
        # We take double the amount and sample to increase variety
        assoc_recs = self.get_recommendations(book_title, num_recommendations=num_recommendations * 2)
        
        final_recs = []
        seen_titles = set([book_title.strip().upper()])

        # Add top associations first (they are specific to this book)
        if assoc_recs is not None and not assoc_recs.empty:
            # Sample from the top association rules to increase variety on each search
            sampled_assoc = assoc_recs.sample(min(len(assoc_recs), num_recommendations))
            for _, row in sampled_assoc.iterrows():
                title = row.get('Titre', row.get('Titre_clean'))
                if title not in seen_titles:
                    final_recs.append({
                        'Titre': title,
                        'Auteur': row.get('Auteur', 'Unknown')
                    })
                    seen_titles.add(title)

        # 2. Top-up with Profile-Based (Clustering) Recommendations for Context
        if len(final_recs) < num_recommendations:
            # Find Titre_clean for the book
            book_info = self.data_loader.books_df[
                (self.data_loader.books_df['Titre'] == book_title) | 
                (self.data_loader.books_df['Titre_clean'] == book_title)
            ]
            
            if not book_info.empty:
                titre_clean = book_info.iloc[0]['Titre_clean']
                
                # Find users who borrowed this book
                users_who_borrowed = self.data_loader.borrowings_df[
                    self.data_loader.borrowings_df['Titre_clean'] == titre_clean
                ]['N° lecteur'].unique()
                
                if len(users_who_borrowed) > 0 and clustering_analysis.df_profiles is not None:
                    # Find common cluster
                    profiles = clustering_analysis.df_profiles[
                        clustering_analysis.df_profiles['N° lecteur'].isin(users_who_borrowed)
                    ]
                    
                    if not profiles.empty:
                        top_cluster = profiles['kmeans_cluster'].mode()[0]
                        similar_users = clustering_analysis.df_profiles[
                            clustering_analysis.df_profiles['kmeans_cluster'] == top_cluster
                        ]['N° lecteur'].tolist()
                        
                        # Find what else cluster users liked
                        cluster_popular = self.data_loader.borrowings_df[
                            (self.data_loader.borrowings_df['N° lecteur'].isin(similar_users)) &
                            (~self.data_loader.borrowings_df['Titre_clean'].isin(seen_titles))
                        ]['Titre_clean'].value_counts().head(num_recommendations * 2).reset_index()
                        
                        if not cluster_popular.empty:
                            # Sample from cluster popular for variety
                            cluster_popular.columns = ['Titre_clean', 'count']
                            to_add_df = cluster_popular.sample(min(len(cluster_popular), num_recommendations - len(final_recs)))
                            
                            for _, row in to_add_df.iterrows():
                                t_clean = row['Titre_clean']
                                meta = self.data_loader.books_df[self.data_loader.books_df['Titre_clean'] == t_clean]
                                if not meta.empty:
                                    info = meta.iloc[0]
                                    final_recs.append({
                                        'Titre': info.get('Titre', t_clean),
                                        'Auteur': self.data_loader.clean_author_name(info.get('Auteur', 'Unknown'))
                                    })
                                else:
                                    final_recs.append({
                                        'Titre': t_clean,
                                        'Auteur': 'Unknown'
                                    })
                                seen_titles.add(t_clean)

        # 3. Final Fallback to popularity
        if len(final_recs) < num_recommendations:
            pop = self.get_popular_recommendations(num_recommendations * 3)
            pop = pop[~pop['Titre'].isin(seen_titles)].sample(min(len(pop), num_recommendations - len(final_recs)))
            for _, row in pop.iterrows():
                final_recs.append(row.to_dict())

        return pd.DataFrame(final_recs).head(num_recommendations)

    def get_rule_statistics(self):
        """Get statistics about association rules"""
        if self.association_rules_df is None or len(self.association_rules_df) == 0:
            return {
                "total_rules": 0,
                "avg_confidence": 0,
                "avg_lift": 0,
                "avg_support": 0,
                "strong_rules": 0,
            }

        stats = {
            "total_rules": len(self.association_rules_df),
            "avg_confidence": self.association_rules_df["confidence"].mean(),
            "avg_lift": self.association_rules_df["lift"].mean(),
            "avg_support": self.association_rules_df["support"].mean(),
            "strong_rules": len(self.association_rules_df[self.association_rules_df["lift"] > 1.5]),
            "max_confidence": self.association_rules_df["confidence"].max(),
            "max_lift": self.association_rules_df["lift"].max(),
            "total_itemsets": len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
        }

        return stats

    def save_model(self):
        """Save trained model to disk"""
        model_data = {
            "frequent_itemsets": self.frequent_itemsets,
            "association_rules": self.association_rules_df,
            "min_support": self.min_support,
            "min_confidence": self.min_confidence,
            "transaction_df": self.transaction_df,
        }

        model_file = self.model_path / "recommendation_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load pre-trained model from disk"""
        model_file = self.model_path / "recommendation_model.pkl"
        if not model_file.exists():
            return False

        try:
            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            self.frequent_itemsets = model_data["frequent_itemsets"]
            self.association_rules_df = model_data["association_rules"]
            self.min_support = model_data["min_support"]
            self.min_confidence = model_data["min_confidence"]
            self.transaction_df = model_data.get("transaction_df", None)

            return True
        except Exception as e:
            return False

    def retrain_model(self, min_support=None, min_confidence=None):
        """Retrain model with new parameters"""
        if min_support is not None:
            self.min_support = min_support
        if min_confidence is not None:
            self.min_confidence = min_confidence

        self.train_model()