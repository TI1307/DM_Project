

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizations:
    """Visualization methods for the book recommendation system"""
    
    def __init__(self, data_loader):
        """Initialize visualizations
        
        Args:
            data_loader: DataLoader instance with borrowings_df and books_df
        """
        self.data_loader = data_loader
        
        # Color schemes
        self.color_primary = '#6b4423'
        self.color_secondary = '#8d6e63'
        self.color_accent = '#a1887f'
        self.color_light = '#d7ccc8'
    
    def show_top_books_chart(self, top_n=10):
        """Display top borrowed books as bar chart"""
        st.markdown(f"### Top {top_n} Most Popular Books")
        
        title_col = 'Titre_clean' if 'Titre_clean' in self.data_loader.borrowings_df.columns else 'Titre'
        top_books = self.data_loader.borrowings_df[title_col].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_books.values,
            y=top_books.index,
            orientation='h',
            labels={'x': 'Number of Checkouts', 'y': ''},
            text=top_books.values,
            color=top_books.values,
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400 + (top_n * 20),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Lato, sans-serif", size=11),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="top_books_chart")
    
    def show_category_distribution(self):
        """Display distribution of books by category (pie chart)"""
        st.markdown("### Book Category Distribution")
        
        # Check topic_fr in books_df
        if 'topic_fr' not in self.data_loader.books_df.columns:
            st.warning("Category information not available")
            return
            
        category_counts = self.data_loader.books_df['topic_fr'].value_counts()
        
        # Filter out generic/uninformative categories for the pie chart
        informed_counts = category_counts[~category_counts.index.isin(['Non classé', 'Autre', 'Sans DDC', ''])]
        informed_counts = informed_counts.head(10)
        
        if len(informed_counts) == 0:
            # Fallback to show whatever we have if filtering left nothing
            informed_counts = category_counts.head(10)
            
        if len(informed_counts) == 0:
            st.warning("No category data available to display")
            return
            
        fig = px.pie(
            values=informed_counts.values,
            names=informed_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Tealgrn_r
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=11, family="Lato, sans-serif"),
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        
        fig.update_layout(
            height=500,
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="category_distribution_pie")
    
    def show_user_category_distribution(self):
        """Display distribution of users by category"""
        st.markdown("### Checkouts by User Type")
        
        if 'Catégorie' not in self.data_loader.borrowings_df.columns:
            st.info("User category information not available")
            return
        
        user_categories = self.data_loader.borrowings_df['Catégorie'].value_counts()
        
        fig = px.bar(
            x=user_categories.index,
            y=user_categories.values,
            labels={'x': '', 'y': 'Number of Checkouts'},
            text=user_categories.values,
            color=user_categories.values,
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400,
            showlegend=False,
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=-45),
            margin=dict(l=20, r=20, t=20, b=80)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="user_category_distribution")
    
    def show_books_per_user_distribution(self):
        """Display distribution of number of books per user with INTEGER x-axis"""
        st.markdown("### Checkout Patterns per User")
        
        books_per_user = self.data_loader.borrowings_df.groupby('N° lecteur')['Titre'].count()
        
        # Create bins for integer values
        max_books = int(books_per_user.max())
        bins = list(range(0, min(max_books + 2, 51)))  # Cap at 50 for readability
        
        # Create histogram data
        hist_data, bin_edges = np.histogram(books_per_user, bins=bins)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bin_edges[:-1],  # Use left edges as x values
            y=hist_data,
            marker_color=self.color_secondary,
            text=hist_data,
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis_title='Number of Books Borrowed',
            yaxis_title='Number of Users',
            showlegend=False,
            height=400,
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(245,240,235,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,  # Force every integer to show
                tickformat='d', # Integer format
                range=[0.5, min(max_books + 1, 30)] # Start from 1 borrowed book
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, key="books_per_user_dist")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{books_per_user.mean():.1f} books")
        with col2:
            st.metric("Median", f"{int(books_per_user.median())} books")
        with col3:
            st.metric("Maximum", f"{int(books_per_user.max())} books")
        with col4:
            st.metric("Minimum", f"{int(books_per_user.min())} books")
    
    def show_top_borrowed_categories(self, top_n=20):
        """Top N Most Borrowed Book Categories"""
        st.markdown(f"### Top {top_n} Most Borrowed Categories")
        
        # Merge with books_df to get topics if not direct
        if 'topic_fr' in self.data_loader.borrowings_df.columns:
            borrowings_by_category = self.data_loader.borrowings_df['topic_fr'].value_counts()
        else:
            merged = self.data_loader.borrowings_df.merge(
                self.data_loader.books_df[['Titre_clean', 'topic_fr']],
                on='Titre_clean',
                how='left'
            )
            borrowings_by_category = merged['topic_fr'].value_counts()
        
        # Filter out generic/unmapped categories for better visualization
        top_categories = borrowings_by_category[~borrowings_by_category.index.isin(['Non classé', 'Autre', 'Sans DDC'])].head(top_n)
        
        # Create horizontal bar chart
        fig = px.bar(
            x=top_categories.values,
            y=top_categories.index,
            orientation='h',
            labels={'x': 'Total Checkouts', 'y': 'Book Category'},
            text=top_categories.values,
            color=top_categories.values,
            color_continuous_scale=[[0, self.color_light], [1, self.color_primary]]
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400 + (top_n * 20),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Lato, sans-serif", size=11),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="top_borrowed_categories")
    
    def show_user_segmentation_by_borrowing_frequency(self):
        """User Segmentation by Borrowing Frequency - Pie chart"""
        st.markdown("### User Activity Levels")
        
        # Group by user to create transactions
        borrowings_transactions = self.data_loader.borrowings_df.groupby("N° lecteur")["Titre_clean"].apply(list).reset_index()
        
        # Calculate number of books per user
        borrowings_transactions['num_books_borrowed'] = borrowings_transactions['Titre_clean'].apply(len)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(borrowings_transactions))
        with col2:
            st.metric("Total Checkouts", len(self.data_loader.borrowings_df))
        with col3:
            st.metric("Avg Books/User", f"{borrowings_transactions['num_books_borrowed'].mean():.2f}")
        
        # Create borrowing categories
        borrowing_categories = pd.cut(
            borrowings_transactions['num_books_borrowed'],
            bins=[0, 1, 2, 3, 5, float('inf')],
            labels=['1 book', '2 books', '3 books', '4-5 books', '6+ books']
        )
        
        category_counts = borrowing_categories.value_counts().sort_index()
        
        # Add user counts to labels
        labels = [f'{label}<br>({count} users)' for label, count in zip(category_counts.index, category_counts.values)]
        
        # Create pie chart
        fig = px.pie(
            values=category_counts.values,
            names=labels,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=12, family="Lato, sans-serif"),
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        
        fig.update_layout(
            height=600,
            font=dict(family="Lato, sans-serif", size=12),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="user_segmentation_pie")
    
    def show_top_authors(self, top_n=10):
        """Display top authors by number of books"""
        st.markdown(f"### Top {top_n} Authors by Number of Books")
        
        if 'Auteur' not in self.data_loader.books_df.columns:
            st.info("Author information not available")
            return
        
        top_authors = self.data_loader.books_df['Auteur'].apply(self.data_loader.clean_author_name).value_counts()
        
        # Remove all variants of Unknown
        unknown_variants = ['Unknown', 'Unknown Author', 'nan', 'None', '', ' ']
        top_authors = top_authors[~top_authors.index.isin(unknown_variants)]
        top_authors = top_authors[top_authors.index.str.len() > 2]
        
        top_authors = top_authors.head(top_n)
        
        fig = px.bar(
            x=top_authors.values,
            y=top_authors.index,
            orientation='h',
            labels={'x': 'Number of Books', 'y': ''},
            text=top_authors.values,
            color=top_authors.values,
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400 + (top_n * 15),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Lato, sans-serif", size=11, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="top_authors_chart")