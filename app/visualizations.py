import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

class Visualizations:
    """Visualization methods for the book recommendation system"""
    
    def __init__(self, data_loader):
        """Initialize visualizations
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
    
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
            labels={'x': 'Number of Borrowings', 'y': 'Book Title'},
            title=f'Top {top_n} Most Borrowed Books',
            color=top_books.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=400 + (top_n * 20),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_category_distribution(self):
        """Display distribution of books by category"""
        if 'topic_fr' not in self.data_loader.books_df.columns:
            st.warning("Category information not available")
            return
        
        category_counts = self.data_loader.books_df['topic_fr'].value_counts().head(15)
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title='Books Distribution by Category (Top 15)',
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_borrowing_duration_distribution(self):
        """Display distribution of borrowing durations"""
        if 'borrowing duration' not in self.data_loader.borrowings_df.columns:
            st.warning("Borrowing duration information not available")
            return
        
        durations = self.data_loader.borrowings_df['borrowing duration'].dropna()
        
        fig = px.histogram(
            durations,
            nbins=30,
            title='Distribution of Borrowing Duration',
            labels={'value': 'Days', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(
            xaxis_title='Borrowing Duration (days)',
            yaxis_title='Number of Borrowings',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Duration", f"{durations.mean():.1f} days")
        with col2:
            st.metric("Median Duration", f"{durations.median():.1f} days")
        with col3:
            st.metric("Max Duration", f"{durations.max():.0f} days")
    
    def show_user_category_distribution(self):
        """Display distribution of users by category"""
        if 'Catégorie' not in self.data_loader.borrowings_df.columns:
            st.warning("User category information not available")
            return
        
        user_categories = self.data_loader.borrowings_df['Catégorie'].value_counts()
        
        fig = px.bar(
            x=user_categories.index,
            y=user_categories.values,
            title='Users by Category',
            labels={'x': 'Category', 'y': 'Number of Borrowings'},
            color=user_categories.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_user_segmentation_pie(self):
        """Display user segmentation by borrowing frequency"""
        if self.data_loader.user_stats is None:
            st.warning("User statistics not available")
            return
        
        segmentation = self.data_loader.user_stats['category'].value_counts()
        
        # Define colors for each category
        colors = {
            'Light Reader': '#3498db',
            'Moderate Reader': '#2ecc71',
            'Heavy Reader': '#e74c3c'
        }
        
        color_sequence = [colors.get(cat, '#95a5a6') for cat in segmentation.index]
        
        fig = px.pie(
            values=segmentation.values,
            names=segmentation.index,
            title='User Segmentation by Borrowing Frequency',
            hole=0.4,
            color_discrete_sequence=color_sequence
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12
        )
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.markdown("### Segmentation Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            light = segmentation.get('Light Reader', 0)
            st.metric("Light Readers", f"{light:,}", "≤ 5 books")
        
        with col2:
            moderate = segmentation.get('Moderate Reader', 0)
            st.metric("Moderate Readers", f"{moderate:,}", "6-15 books")
        
        with col3:
            heavy = segmentation.get('Heavy Reader', 0)
            st.metric("Heavy Readers", f"{heavy:,}", "> 15 books")
    
    def show_borrowing_by_student_year(self):
        """Display borrowing patterns by student year/user type"""
        if 'Catégorie' not in self.data_loader.borrowings_df.columns:
            st.warning("User category information not available")
            return
        
        # Get borrowing counts by category
        category_borrowings = self.data_loader.borrowings_df.groupby('Catégorie').size().reset_index(name='Borrowings')
        category_borrowings = category_borrowings.sort_values('Borrowings', ascending=True)
        
        fig = px.bar(
            category_borrowings,
            x='Borrowings',
            y='Catégorie',
            orientation='h',
            title='Total Borrowings by User Category',
            labels={'Borrowings': 'Number of Borrowings', 'Catégorie': 'User Category'},
            color='Borrowings',
            color_continuous_scale='Teal'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_books_per_user_distribution(self):
        """Display distribution of number of books per user"""
        books_per_user = self.data_loader.borrowings_df.groupby('N° lecteur')['Titre'].count()
        
        fig = px.histogram(
            books_per_user,
            nbins=50,
            title='Distribution of Books Borrowed per User',
            labels={'value': 'Number of Books', 'count': 'Number of Users'},
            color_discrete_sequence=['#2ecc71']
        )
        
        fig.update_layout(
            xaxis_title='Number of Books Borrowed',
            yaxis_title='Number of Users',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Books/User", f"{books_per_user.mean():.1f}")
        with col2:
            st.metric("Median Books/User", f"{books_per_user.median():.1f}")
        with col3:
            st.metric("Max Books/User", f"{books_per_user.max():.0f}")
    
    def show_top_authors(self, top_n=10):
        """Display top authors by number of books
        
        Args:
            top_n: Number of top authors to display
        """
        if 'Auteur_merged1' not in self.data_loader.books_df.columns:
            st.warning("Author information not available")
            return
        
        top_authors = self.data_loader.books_df['Auteur_merged1'].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_authors.values,
            y=top_authors.index,
            orientation='h',
            title=f'Top {top_n} Authors by Number of Books',
            labels={'x': 'Number of Books', 'y': 'Author'},
            color=top_authors.values,
            color_continuous_scale='Oranges'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_correlation_heatmap(self, top_n=20):
        """Display correlation heatmap of top books
        
        Args:
            top_n: Number of top books to include
        """
        # Get top books
        title_col = 'Titre_clean' if 'Titre_clean' in self.data_loader.borrowings_df.columns else 'Titre'
        top_books = self.data_loader.borrowings_df[title_col].value_counts().head(top_n).index
        
        # Filter user-book matrix to only include top books that exist in the matrix
        available_books = [book for book in top_books if book in self.data_loader.user_book_matrix.columns]
        
        if not available_books:
            st.warning("No matching books found in the user-book matrix")
            return
        
        filtered_matrix = self.data_loader.user_book_matrix[available_books]
        
        # Calculate correlation
        correlation = filtered_matrix.corr()
        
        fig = px.imshow(
            correlation,
            title=f'Book Correlation Heatmap (Top {len(available_books)} Books)',
            labels=dict(color="Correlation"),
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_association_rules_metrics(self, recommendation_engine):
        """Display association rules quality metrics
        
        Args:
            recommendation_engine: RecommendationEngine instance
        """
        stats = recommendation_engine.get_rule_statistics()
        
        st.markdown("### Association Rules Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rules", f"{stats['total_rules']:,}")
        
        with col2:
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.3f}")
        
        with col3:
            st.metric("Avg Lift", f"{stats['avg_lift']:.2f}")
        
        with col4:
            st.metric("Strong Rules", f"{stats['strong_rules']:,}")
        
        # Show distribution of confidence and lift
        if recommendation_engine.association_rules_df is not None and len(recommendation_engine.association_rules_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_conf = px.histogram(
                    recommendation_engine.association_rules_df,
                    x='confidence',
                    nbins=30,
                    title='Distribution of Confidence',
                    labels={'confidence': 'Confidence', 'count': 'Frequency'},
                    color_discrete_sequence=['#3498db']
                )
                fig_conf.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with col2:
                fig_lift = px.histogram(
                    recommendation_engine.association_rules_df,
                    x='lift',
                    nbins=30,
                    title='Distribution of Lift',
                    labels={'lift': 'Lift', 'count': 'Frequency'},
                    color_discrete_sequence=['#e74c3c']
                )
                fig_lift.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_lift, use_container_width=True)
    
    def show_top_association_rules(self, recommendation_engine, top_n=15):
        """Display top association rules
        
        Args:
            recommendation_engine: RecommendationEngine instance
            top_n: Number of top rules to display
        """
        st.markdown("### Top Association Rules")
        
        sort_option = st.selectbox(
            "Sort by:",
            ['confidence', 'lift', 'support'],
            index=0
        )
        
        top_rules = recommendation_engine.get_top_rules(top_n=top_n, sort_by=sort_option)
        
        if not top_rules.empty:
            # Display as a styled dataframe
            st.dataframe(
                top_rules.style.format({
                    'support': '{:.4f}',
                    'confidence': '{:.4f}',
                    'lift': '{:.2f}',
                    'conviction': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Visualize top rules
            fig = px.scatter(
                top_rules,
                x='confidence',
                y='lift',
                size='support',
                hover_data=['antecedents_str', 'consequents_str'],
                title=f'Top {top_n} Association Rules',
                labels={'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'},
                color='lift',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No association rules available")
    
    def show_network_graph(self, recommendation_engine, book_title, max_connections=10):
        """Display network graph of related books
        
        Args:
            recommendation_engine: RecommendationEngine instance
            book_title: Central book title
            max_connections: Maximum number of connections to show
        """
        if recommendation_engine.association_rules_df is None or len(recommendation_engine.association_rules_df) == 0:
            st.warning("No association rules available for network visualization")
            return
        
        # Get book associations
        analysis = recommendation_engine.analyze_book_associations(book_title)
        
        if analysis is None or analysis['total_associations'] == 0:
            st.info(f"No associations found for '{book_title}'")
            return
        
        # Create network graph
        G = nx.Graph()
        G.add_node(book_title, node_type='center')
        
        # Add nodes from antecedent rules (books that lead to this book)
        if not analysis['antecedent_rules'].empty:
            for idx, rule in analysis['antecedent_rules'].head(max_connections).iterrows():
                for consequent in rule['consequents']:
                    if consequent != book_title:
                        G.add_node(consequent, node_type='recommended')
                        G.add_edge(book_title, consequent, 
                                 weight=rule['confidence'],
                                 lift=rule['lift'])
        
        # Add nodes from consequent rules (books often borrowed with this book)
        if not analysis['consequent_rules'].empty:
            for idx, rule in analysis['consequent_rules'].head(max_connections).iterrows():
                for antecedent in rule['antecedents']:
                    if antecedent != book_title and antecedent not in G.nodes():
                        G.add_node(antecedent, node_type='related')
                        G.add_edge(book_title, antecedent,
                                 weight=rule['confidence'],
                                 lift=rule['lift'])
        
        # Get positions for nodes
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Extract node positions
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Color based on node type
            if G.nodes[node]['node_type'] == 'center':
                node_color.append('#e74c3c')
                node_size.append(30)
            elif G.nodes[node]['node_type'] == 'recommended':
                node_color.append('#3498db')
                node_size.append(20)
            else:
                node_color.append('#2ecc71')
                node_size.append(15)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line_width=2
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Book Association Network for "{book_title}"',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Books Recommended From This", analysis['as_antecedent'])
        with col2:
            st.metric("Books Often Borrowed Together", analysis['as_consequent'])