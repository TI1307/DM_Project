import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class Visualizations:
    """Visualization methods for the book recommendation system"""
    
    def __init__(self, data_loader):
        """Initialize visualizations
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
        # Brown color scheme
        self.color_primary = '#6b4423'
        self.color_secondary = '#8d6e63'
        self.color_accent = '#a1887f'
        self.color_light = '#d7ccc8'
    
    def show_top_books_chart(self, top_n=10):
        """Display top borrowed books as bar chart
        
        Args:
            top_n: Number of top books to display
        """
        title_col = 'Titre_clean' if 'Titre_clean' in self.data_loader.borrowings_df.columns else 'Titre'
        top_books = self.data_loader.borrowings_df[title_col].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_books.values,
            y=top_books.index,
            orientation='h',
            labels={'x': 'Number of Borrowings', 'y': ''},
            color=top_books.values,
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_layout(
            height=400 + (top_n * 20),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_category_distribution(self):
        """Display distribution of books by category"""
        if 'topic_fr' not in self.data_loader.books_df.columns:
            st.warning("Category information not available")
            return
        
        category_counts = self.data_loader.books_df['topic_fr'].value_counts().head(10)
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.deep
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_user_category_distribution(self):
        """Display distribution of users by category"""
        if 'Catégorie' not in self.data_loader.borrowings_df.columns:
            st.info("User category information not available")
            return
        
        user_categories = self.data_loader.borrowings_df['Catégorie'].value_counts()
        
        fig = px.bar(
            x=user_categories.index,
            y=user_categories.values,
            labels={'x': '', 'y': 'Number of Borrowings'},
            color=user_categories.values,
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=-45),
            margin=dict(l=20, r=20, t=20, b=80)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_user_segmentation_pie(self):
        """Display user segmentation by borrowing frequency"""
        if self.data_loader.user_stats is None:
            st.info("User statistics not available")
            return
        
        segmentation = self.data_loader.user_stats['category'].value_counts()
        
        # Custom colors for segmentation
        colors = ['#d7ccc8', '#8d6e63', '#6b4423']  # Light to dark brown
        
        fig = px.pie(
            values=segmentation.values,
            names=segmentation.index,
            hole=0.5,
            color_discrete_sequence=colors
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=13, family="Lato, sans-serif", color='white'),
            marker=dict(line=dict(color='#ffffff', width=3))
        )
        
        fig.update_layout(
            height=450,
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_borrowing_by_student_year(self):
        """Display borrowing patterns by student year/user type"""
        if 'Catégorie' not in self.data_loader.borrowings_df.columns:
            st.info("User category information not available")
            return
        
        category_borrowings = self.data_loader.borrowings_df.groupby('Catégorie').size().reset_index(name='Borrowings')
        category_borrowings = category_borrowings.sort_values('Borrowings', ascending=True)
        
        fig = px.bar(
            category_borrowings,
            x='Borrowings',
            y='Catégorie',
            orientation='h',
            labels={'Borrowings': 'Number of Borrowings', 'Catégorie': ''},
            color='Borrowings',
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_books_per_user_distribution(self):
        """Display distribution of number of books per user"""
        books_per_user = self.data_loader.borrowings_df.groupby('N° lecteur')['Titre'].count()
        
        fig = px.histogram(
            books_per_user,
            nbins=40,
            labels={'value': 'Number of Books', 'count': 'Number of Readers'},
            color_discrete_sequence=[self.color_secondary]
        )
        
        fig.update_layout(
            xaxis_title='Number of Books Borrowed',
            yaxis_title='Number of Readers',
            showlegend=False,
            height=400,
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(245,240,235,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{books_per_user.mean():.1f} books")
        with col2:
            st.metric("Median", f"{books_per_user.median():.0f} books")
        with col3:
            st.metric("Maximum", f"{books_per_user.max():.0f} books")
    
    def show_top_authors(self, top_n=10):
        """Display top authors by number of books
        
        Args:
            top_n: Number of top authors to display
        """
        if 'Auteur' not in self.data_loader.books_df.columns:
            st.info("Author information not available")
            return
        
        top_authors = self.data_loader.books_df['Auteur'].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_authors.values,
            y=top_authors.index,
            orientation='h',
            labels={'x': 'Number of Books', 'y': ''},
            color=top_authors.values,
            color_continuous_scale=[[0, self.color_light], [0.5, self.color_secondary], [1, self.color_primary]]
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(family="Lato, sans-serif", size=12, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_library_collection_overview(self):
        """Display overview of entire library collection"""
        stats = self.data_loader.get_statistics()
        
        # Create a simple bar chart showing borrowed vs unborrowed
        categories = ['Books in Circulation', 'Books Available']
        values = [stats['borrowed_books'], stats['unborrowed_books']]
        
        fig = px.bar(
            x=categories,
            y=values,
            labels={'x': '', 'y': 'Number of Books'},
            color=values,
            color_continuous_scale=[[0, self.color_accent], [1, self.color_primary]]
        )
        
        fig.update_layout(
            height=350,
            showlegend=False,
            font=dict(family="Lato, sans-serif", size=13, color=self.color_primary),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        fig.update_traces(text=values, textposition='outside', textfont=dict(size=16, color=self.color_primary))
        
        st.plotly_chart(fig, use_container_width=True)