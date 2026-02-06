"""
Clustering Analysis Module

This module provides clustering capabilities for library user analysis.
Implements K-Means, DBSCAN, and Hierarchical clustering algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Define book category mapping with grouped categories
CATEGORY_MAPPING = {
    # Algebra
    '512': 'Math_Algebra',

    # Analysis / Calculus
    '515': 'Math_Analysis',

    # Probability & Statistics
    '519': 'Math_Statistics',

    # General Mathematics
    '510': 'Math_General',
    '511': 'Math_General',
    '518': 'Math_General',

    # Computer Science
    '004': 'Computer_Science',
    '005': 'Computer_Science',
    '006': 'Computer_Science',
    '681': 'Computer_Science',

    # Engineering
    '621': 'Engineering',

    # Everything else → OTHER
    '150': 'Other',   # Psychology
    '230': 'Other',   # Religion
    '350': 'Other',   # Public admin
    '380': 'Other',   # Commerce
    '570': 'Other',   # Life sciences
    '610': 'Other',   # Medicine
    '611': 'Other',
    '616': 'Other',
    '658': 'Other',
    '808': 'Other'
}


def extract_category_from_cote(cote):
    """Extract and map category code from Cote to grouped categories"""
    if pd.isna(cote):
        return 'Other'
    
    cote_str = str(cote).strip()
    
    # Extract first 3 digits and map to grouped category
    for key in CATEGORY_MAPPING.keys():
        if cote_str.startswith(key):
            return CATEGORY_MAPPING[key]
    
    return 'Other'


def create_user_profiles(df):
    """Create user profiles with grouped category percentages"""
    profiles = []
    
    # Get all unique grouped categories
    all_categories = df['book_category'].unique()
    
    for borrower_id in df['N° lecteur'].unique():
        user_data = df[df['N° lecteur'] == borrower_id]
        
        profile = {
            'N° lecteur': borrower_id,
            'Nom': user_data['Nom'].iloc[0],
            'Prénom': user_data['Prénom'].iloc[0],
            'Catégorie': user_data['Catégorie'].iloc[0],
            'total_borrowed': len(user_data),
            'avg_duration': user_data['borrowing duration'].mean(),
            'diversity_score': user_data['book_category'].nunique()
        }
        
        # Calculate percentage for each grouped category
        category_counts = user_data['book_category'].value_counts()
        total_books = len(user_data)
        
        for category in all_categories:
            count = category_counts.get(category, 0)
            profile[f'{category}_pct'] = (count / total_books) * 100
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


class ClusteringAnalysis:
    """Clustering analysis for users based on borrowing behavior"""
    
    def __init__(self, data_loader):
        """Initialize clustering analysis
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        self.df_borrowing = None
        self.df_profiles = None
        self.X_scaled = None
        self.X_pca = None
        self.pca = None
        self.scaler = StandardScaler()
        
        # Clustering results
        self.kmeans_labels = None
        self.dbscan_labels = None
        self.hierarchical_labels = None
        
        # Optimal parameters
        self.optimal_k = None
        self.optimal_eps = None
        self.optimal_n_hierarchical = None
        
        # Metrics
        self.kmeans_silhouette = None
        self.dbscan_silhouette = None
        self.hierarchical_silhouette = None
        self.n_clusters_dbscan = None
        self.n_noise_dbscan = None
    
    def prepare_data(self):
        """Prepare borrowing data with book categories"""
        # Get borrowing data
        self.df_borrowing = self.data_loader.borrowings_df.copy()
        
        # Apply category extraction
        self.df_borrowing['book_category'] = self.df_borrowing['Cote'].apply(extract_category_from_cote)
        
        return self.df_borrowing
    
    def create_profiles(self):
        """Create user profiles with features"""
        if self.df_borrowing is None:
            self.prepare_data()
        
        # Create user profiles
        self.df_profiles = create_user_profiles(self.df_borrowing)
        
        return self.df_profiles
    
    def prepare_features(self):
        """Prepare feature matrix for clustering"""
        if self.df_profiles is None:
            self.create_profiles()
        
        # One-hot encode reader category
        reader_categories_dummies = pd.get_dummies(self.df_profiles['Catégorie'], prefix='reader_cat')
        
        # Select percentage columns (grouped categories)
        pct_columns = [col for col in self.df_profiles.columns if col.endswith('_pct')]
        
        # Behavioral features
        behavioral_features = [
            'total_borrowed',
            'avg_duration',
            'diversity_score'
        ]
        
        # Create feature matrix
        X_behavioral = self.df_profiles[behavioral_features + pct_columns].copy()
        X_full = pd.concat([X_behavioral, reader_categories_dummies], axis=1)
        
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(X_full)
        
        # Apply PCA for visualization
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        return self.X_scaled, self.X_pca
    
    def find_optimal_k(self, k_range=None):
        """Find optimal number of clusters for K-Means using elbow method and silhouette score"""
        if k_range is None:
            k_range = range(2, 11)
        
        if self.X_scaled is None:
            self.prepare_features()
        
        inertias = []
        silhouette_scores_list = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores_list.append(silhouette_score(self.X_scaled, labels))
        
        # Create visualization
        fig = go.Figure()
        
        # Elbow plot
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            yaxis='y1'
        ))
        
        # Silhouette plot
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores_list,
            mode='lines+markers',
            name='Silhouette Score',
            yaxis='y2',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='K-Means Optimization Metrics',
            xaxis=dict(title='Number of Clusters (k)'),
            yaxis=dict(title='Inertia', side='left'),
            yaxis2=dict(title='Silhouette Score', side='right', overlaying='y'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics table
        metrics_df = pd.DataFrame({
            'k': list(k_range),
            'Inertia': inertias,
            'Silhouette': silhouette_scores_list
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        return silhouette_scores_list
    
    def cluster_kmeans(self, n_clusters=4):
        """Apply K-Means clustering"""
        if self.X_scaled is None:
            self.prepare_features()
        
        self.optimal_k = n_clusters
        
        # Apply K-Means
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans_labels = kmeans_model.fit_predict(self.X_scaled)
        
        # Add cluster labels to profiles
        self.df_profiles['kmeans_cluster'] = self.kmeans_labels
        
        # Calculate silhouette score
        self.kmeans_silhouette = silhouette_score(self.X_scaled, self.kmeans_labels)
        
        return self.kmeans_labels
    
    def find_optimal_eps_dbscan(self, k=8):
        """Find optimal eps for DBSCAN using k-distance graph"""
        if self.X_scaled is None:
            self.prepare_features()
        
        # Calculate k-distance graph
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(self.X_scaled)
        distances, indices = neighbors_fit.kneighbors(self.X_scaled)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Plot k-distance graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(distances))),
            y=distances,
            mode='lines',
            name=f'{k}-NN Distance'
        ))
        
        fig.update_layout(
            title='K-distance Graph for DBSCAN eps Selection',
            xaxis_title='Data Points (sorted)',
            yaxis_title=f'{k}-NN Distance',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        suggested_eps_low = distances[int(len(distances)*0.9)]
        suggested_eps_high = distances[int(len(distances)*0.95)]
        
        st.info(f"Suggested eps range: {suggested_eps_low:.2f} - {suggested_eps_high:.2f}")
        
        return suggested_eps_low, suggested_eps_high
    
    def test_dbscan_params(self, eps_values=None, min_samples=8):
        """Test different eps values for DBSCAN"""
        if eps_values is None:
            eps_values = [3.90, 4.5, 5.0, 5.7, 6.2]
        
        if self.X_scaled is None:
            self.prepare_features()
        
        results = []
        best_eps = None
        best_score = -1
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(self.X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            result = {
                'eps': eps,
                'clusters': n_clusters,
                'noise_points': n_noise,
                'noise_pct': f"{n_noise/len(labels)*100:.1f}%",
                'silhouette': 'N/A'
            }
            
            if n_clusters > 1 and n_noise < len(labels) * 0.3:
                valid_labels = labels[labels != -1]
                valid_data = self.X_scaled[labels != -1]
                
                if len(set(valid_labels)) > 1:
                    score = silhouette_score(valid_data, valid_labels)
                    result['silhouette'] = f"{score:.3f}"
                    
                    if score > best_score:
                        best_score = score
                        best_eps = eps
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        if best_eps is not None:
            st.success(f"Best eps: {best_eps} (Silhouette: {best_score:.3f})")
        
        return best_eps
    
    def cluster_dbscan(self, eps=5.7, min_samples=8):
        """Apply DBSCAN clustering"""
        if self.X_scaled is None:
            self.prepare_features()
        
        self.optimal_eps = eps
        
        # Apply DBSCAN
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = dbscan_model.fit_predict(self.X_scaled)
        
        # Add cluster labels to profiles
        self.df_profiles['dbscan_cluster'] = self.dbscan_labels
        
        self.n_clusters_dbscan = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        self.n_noise_dbscan = list(self.dbscan_labels).count(-1)
        
        # Calculate silhouette score (excluding noise)
        if self.n_clusters_dbscan > 1:
            valid_labels = self.dbscan_labels[self.dbscan_labels != -1]
            valid_data = self.X_scaled[self.dbscan_labels != -1]
            
            if len(set(valid_labels)) > 1:
                self.dbscan_silhouette = silhouette_score(valid_data, valid_labels)
        
        return self.dbscan_labels
    
    def plot_dendrogram(self):
        """Plot hierarchical clustering dendrogram"""
        if self.X_scaled is None:
            self.prepare_features()
        
        # Create linkage matrix
        linkage_matrix = linkage(self.X_scaled, method='ward')
        
        # This would need matplotlib for full dendrogram
        # For Streamlit, we'll show a message
        st.info("Dendrogram visualization requires matplotlib. Use find_optimal_hierarchical() to determine optimal clusters.")
        
        return linkage_matrix
    
    def find_optimal_hierarchical(self, n_clusters_range=None):
        """Find optimal number of clusters for Hierarchical clustering"""
        if n_clusters_range is None:
            n_clusters_range = range(2, 11)
        
        if self.X_scaled is None:
            self.prepare_features()
        
        silhouette_scores_list = []
        
        for n in n_clusters_range:
            hierarchical = AgglomerativeClustering(n_clusters=n, linkage='ward')
            labels = hierarchical.fit_predict(self.X_scaled)
            silhouette_scores_list.append(silhouette_score(self.X_scaled, labels))
        
        # Plot metrics
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(n_clusters_range),
            y=silhouette_scores_list,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='Hierarchical Clustering - Silhouette Score',
            xaxis_title='Number of Clusters',
            yaxis_title='Silhouette Score',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics table
        metrics_df = pd.DataFrame({
            'n_clusters': list(n_clusters_range),
            'Silhouette': silhouette_scores_list
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        return silhouette_scores_list
    
    def cluster_hierarchical(self, n_clusters=6):
        """Apply Hierarchical clustering"""
        if self.X_scaled is None:
            self.prepare_features()
        
        self.optimal_n_hierarchical = n_clusters
        
        # Apply Hierarchical Clustering
        hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.hierarchical_labels = hierarchical_model.fit_predict(self.X_scaled)
        
        # Add cluster labels to profiles
        self.df_profiles['hierarchical_cluster'] = self.hierarchical_labels
        
        # Calculate silhouette score
        self.hierarchical_silhouette = silhouette_score(self.X_scaled, self.hierarchical_labels)
        
        return self.hierarchical_labels
    
    def visualize_pca_by_reader_category(self):
        """Visualize user profiles in PCA space colored by reader category"""
        if self.X_pca is None:
            self.prepare_features()
        
        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            'PC1': self.X_pca[:, 0],
            'PC2': self.X_pca[:, 1],
            'Reader_Category': self.df_profiles['Catégorie'].values,
            'Total_Books': self.df_profiles['total_borrowed'],
            'Avg_Duration': self.df_profiles['avg_duration']
        })
        
        fig = px.scatter(
            viz_df,
            x='PC1',
            y='PC2',
            color='Reader_Category',
            title='User Profiles in PCA Space (Colored by Reader Category)',
            hover_data=['Total_Books', 'Avg_Duration'],
            labels={
                'PC1': f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)'
            },
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def visualize_clusters(self, algorithm='kmeans'):
        """Visualize clustering results in PCA space"""
        if self.X_pca is None:
            self.prepare_features()
        
        if algorithm == 'kmeans' and self.kmeans_labels is not None:
            labels = self.kmeans_labels
            title = f'K-Means Clustering Results (k={self.optimal_k})'
            color_label = 'Cluster'
        elif algorithm == 'dbscan' and self.dbscan_labels is not None:
            labels = self.dbscan_labels
            title = f'DBSCAN Clustering Results (eps={self.optimal_eps})'
            color_label = 'Cluster (-1 = noise)'
        elif algorithm == 'hierarchical' and self.hierarchical_labels is not None:
            labels = self.hierarchical_labels
            title = f'Hierarchical Clustering Results (n={self.optimal_n_hierarchical})'
            color_label = 'Cluster'
        else:
            st.warning(f"No {algorithm} clustering results available")
            return
        
        # Create visualization
        viz_df = pd.DataFrame({
            'PC1': self.X_pca[:, 0],
            'PC2': self.X_pca[:, 1],
            'Cluster': labels.astype(str),
            'Total_Books': self.df_profiles['total_borrowed'],
            'Avg_Duration': self.df_profiles['avg_duration']
        })
        
        fig = px.scatter(
            viz_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=title,
            hover_data=['Total_Books', 'Avg_Duration'],
            labels={
                'PC1': f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)',
                'Cluster': color_label
            },
            height=600,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def compare_algorithms(self):
        """Compare all three clustering algorithms"""
        if not all([self.kmeans_labels is not None, 
                   self.dbscan_labels is not None, 
                   self.hierarchical_labels is not None]):
            st.warning("Please run all three clustering algorithms first")
            return
        
        # Create comparison visualization with subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'K-Means (k={self.optimal_k})',
                f'DBSCAN (eps={self.optimal_eps})',
                f'Hierarchical (n={self.optimal_n_hierarchical})'
            )
        )
        
        # K-Means
        fig.add_trace(
            go.Scatter(
                x=self.X_pca[:, 0],
                y=self.X_pca[:, 1],
                mode='markers',
                marker=dict(color=self.kmeans_labels, colorscale='Viridis', size=8),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # DBSCAN
        fig.add_trace(
            go.Scatter(
                x=self.X_pca[:, 0],
                y=self.X_pca[:, 1],
                mode='markers',
                marker=dict(color=self.dbscan_labels, colorscale='Viridis', size=8),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Hierarchical
        fig.add_trace(
            go.Scatter(
                x=self.X_pca[:, 0],
                y=self.X_pca[:, 1],
                mode='markers',
                marker=dict(color=self.hierarchical_labels, colorscale='Viridis', size=8),
                showlegend=False
            ),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text=f"PC1 ({self.pca.explained_variance_ratio_[0]:.2%})")
        fig.update_yaxes(title_text=f"PC2 ({self.pca.explained_variance_ratio_[1]:.2%})")
        
        fig.update_layout(height=500, title_text="Clustering Algorithms Comparison")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison table
        comparison_data = {
            'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical'],
            'N_Clusters': [self.optimal_k, self.n_clusters_dbscan, self.optimal_n_hierarchical],
            'Silhouette': [
                f"{self.kmeans_silhouette:.3f}",
                f"{self.dbscan_silhouette:.3f}" if self.dbscan_silhouette else 'N/A',
                f"{self.hierarchical_silhouette:.3f}"
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        st.markdown("### Performance Comparison")
        st.dataframe(df_comparison, use_container_width=True)
    
    def analyze_cluster_characteristics(self, algorithm='kmeans'):
        """Analyze and display cluster characteristics"""
        if algorithm == 'kmeans' and self.kmeans_labels is not None:
            cluster_col = 'kmeans_cluster'
            n_clusters = self.optimal_k
            title = "K-MEANS CLUSTER ANALYSIS"
        elif algorithm == 'dbscan' and self.dbscan_labels is not None:
            cluster_col = 'dbscan_cluster'
            unique_clusters = [c for c in self.df_profiles[cluster_col].unique() if c != -1]
            n_clusters = len(unique_clusters)
            title = "DBSCAN CLUSTER ANALYSIS"
        elif algorithm == 'hierarchical' and self.hierarchical_labels is not None:
            cluster_col = 'hierarchical_cluster'
            n_clusters = self.optimal_n_hierarchical
            title = "HIERARCHICAL CLUSTER ANALYSIS"
        else:
            st.warning(f"No {algorithm} clustering results available")
            return
        
        st.markdown(f"### {title}")
        
        # Determine cluster IDs to analyze
        if algorithm == 'dbscan':
            cluster_ids = sorted([c for c in self.df_profiles[cluster_col].unique() if c != -1])
        else:
            cluster_ids = range(n_clusters)
        
        for cluster_id in cluster_ids:
            with st.expander(f"Cluster {cluster_id}"):
                cluster_data = self.df_profiles[self.df_profiles[cluster_col] == cluster_id]
                
                # Basic stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Users", len(cluster_data))
                with col2:
                    st.metric("% of Total", f"{len(cluster_data)/len(self.df_profiles)*100:.1f}%")
                with col3:
                    st.metric("Avg Books", f"{cluster_data['total_borrowed'].mean():.1f}")
                with col4:
                    st.metric("Avg Duration", f"{cluster_data['avg_duration'].mean():.1f} days")
                
                # Reader category distribution
                st.markdown("**Borrower Category Distribution:**")
                cat_dist = cluster_data['Catégorie'].value_counts()
                st.dataframe(cat_dist, use_container_width=True)
                
                # Book category preferences
                st.markdown("**Top Book Category Preferences:**")
                category_cols = [col for col in cluster_data.columns if col.endswith('_pct')]
                avg_preferences = cluster_data[category_cols].mean().sort_values(ascending=False).head(5)
                
                pref_df = pd.DataFrame({
                    'Category': [cat.replace('_pct', '') for cat in avg_preferences.index],
                    'Percentage': [f"{val:.1f}%" for val in avg_preferences.values]
                })
                st.dataframe(pref_df, use_container_width=True)
                
                # Common books borrowed
                cluster_user_ids = cluster_data['N° lecteur'].tolist()
                cluster_borrowings = self.df_borrowing[self.df_borrowing['N° lecteur'].isin(cluster_user_ids)]
                
                st.markdown("**Common Book Categories Borrowed:**")
                top_books = cluster_borrowings['book_category'].value_counts().head(5)
                books_df = pd.DataFrame({
                    'Category': top_books.index,
                    'Count': top_books.values
                })
                st.dataframe(books_df, use_container_width=True)
        
        # Analyze noise points for DBSCAN
        if algorithm == 'dbscan' and self.n_noise_dbscan > 0:
            with st.expander("Noise Points (Cluster -1)"):
                noise_data = self.df_profiles[self.df_profiles[cluster_col] == -1]
                st.metric("Users", f"{len(noise_data)} ({len(noise_data)/len(self.df_profiles)*100:.1f}%)")
                st.info("These are outlier users with unique borrowing patterns.")
    
    def save_results(self, output_path='../data/clustering_result/'):
        """Save clustering results to CSV files"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save user profiles with all cluster assignments
        profiles_path = os.path.join(output_path, 'user_profiles_clustered.csv')
        self.df_profiles.to_csv(profiles_path, index=False)
        
        # Save borrowing data with categories
        borrowing_path = os.path.join(output_path, 'borrowing_with_categories.csv')
        self.df_borrowing.to_csv(borrowing_path, index=False)
        
        st.success(f"Results saved to {output_path}")
        st.write(f"- User profiles: {profiles_path}")
        st.write(f"- Borrowing data: {borrowing_path}")
    
    def show_summary(self):
        """Display final summary of clustering analysis"""
        st.markdown("## FINAL SUMMARY")
        st.markdown("---")
        
        st.markdown("### 1. DATA OVERVIEW")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Borrowers", self.df_profiles.shape[0])
        with col2:
            st.metric("Total Borrowings", len(self.df_borrowing))
        with col3:
            st.metric("Book Categories", len(self.df_borrowing['book_category'].unique()))
        
        st.markdown("### 2. CLUSTERING RESULTS")
        
        if self.kmeans_labels is not None:
            st.write(f"**K-Means:** {self.optimal_k} clusters")
        
        if self.dbscan_labels is not None:
            st.write(f"**DBSCAN:** {self.n_clusters_dbscan} clusters + {self.n_noise_dbscan} noise points")
        
        if self.hierarchical_labels is not None:
            st.write(f"**Hierarchical:** {self.optimal_n_hierarchical} clusters")