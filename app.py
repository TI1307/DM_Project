import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from recommendation_engine import RecommendationEngine
from visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="Library Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    with st.spinner("Loading data..."):
        st.session_state.data_loader = DataLoader()
        st.session_state.recommendation_engine = RecommendationEngine(
            st.session_state.data_loader,
            min_support=0.005,
            min_confidence=0.3
        )
        st.session_state.visualizations = Visualizations(st.session_state.data_loader)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/book.png", width=150)
    st.title("📚 Navigation")
    
    page = st.radio(
        "Select Page:",
        [
            "🏠 Home",
            "🔍 Book Recommendations",
            "👤 User Recommendations",
            "📊 Analytics Dashboard",
            "📈 Association Rules",
            "👥 User Segmentation",
            "⚙️ Model Settings"
        ]
    )
    
    st.markdown("---")
    
    # Display quick stats
    stats = st.session_state.data_loader.get_statistics()
    st.markdown("### Quick Stats")
    st.metric("Total Books", f"{stats['total_books']:,}")
    st.metric("Total Users", f"{stats['unique_users']:,}")
    st.metric("Total Borrowings", f"{stats['total_borrowings']:,}")

# Main content
if page == "🏠 Home":
    st.markdown('<div class="main-header">📚 Library Book Recommendation System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Library Book Recommendation System**! This application uses advanced 
    association rule mining to provide personalized book recommendations based on borrowing patterns.
    
    ### 🎯 Features:
    - **Book-to-Book Recommendations**: Find similar books based on borrowing patterns
    - **User-Personalized Recommendations**: Get tailored suggestions for individual users
    - **Advanced Analytics**: Explore borrowing trends and patterns
    - **Association Rules Analysis**: Understand book relationships and correlations
    - **User Segmentation**: Analyze user reading behaviors
    """)
    
    st.markdown("---")
    
    # Display key metrics
    st.markdown("### 📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Books", f"{stats['total_books']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Users", f"{stats['unique_users']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Borrowings", f"{stats['total_borrowings']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Books/User", f"{stats['avg_borrowings_per_user']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show popular books
    st.markdown("### 🔥 Most Popular Books")
    popular_books = st.session_state.data_loader.get_popular_books(10)
    st.dataframe(popular_books, use_container_width=True, height=400)

elif page == "🔍 Book Recommendations":
    st.markdown('<div class="main-header">🔍 Book Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Get book recommendations based on a specific book. The system uses association rules 
    to find books that are frequently borrowed together.
    """)
    
    # Book selection
    title_col = 'Titre_clean' if 'Titre_clean' in st.session_state.data_loader.borrowings_df.columns else 'Titre'
    all_books = sorted(st.session_state.data_loader.borrowings_df[title_col].unique())
    
    selected_book = st.selectbox(
        "Select a book:",
        all_books,
        index=0
    )
    
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10
    )
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = st.session_state.recommendation_engine.get_recommendations(
                selected_book,
                num_recommendations=num_recommendations
            )
            
            if recommendations is not None and len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} recommendations!")
                
                # Display recommendations
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{idx + 1}. {row['Titre']}**")
                            if 'Auteur_merged1' in row and pd.notna(row['Auteur_merged1']):
                                st.markdown(f"*Author: {row['Auteur_merged1']}*")
                            if 'topic_fr' in row and pd.notna(row['topic_fr']):
                                st.markdown(f"📑 Category: {row['topic_fr']}")
                        
                        with col2:
                            st.metric("Confidence", f"{row['confidence']:.3f}")
                            if 'lift' in row:
                                st.metric("Lift", f"{row['lift']:.2f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Show network visualization
                st.markdown("---")
                st.markdown("### 🕸️ Book Association Network")
                st.session_state.visualizations.show_network_graph(
                    st.session_state.recommendation_engine,
                    selected_book,
                    max_connections=10
                )
            else:
                st.warning("No recommendations found. Try selecting a different book.")

elif page == "👤 User Recommendations":
    st.markdown('<div class="main-header">👤 User Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Get personalized recommendations for a specific user based on their borrowing history.
    """)
    
    # User selection
    all_users = sorted(st.session_state.data_loader.borrowings_df['N° lecteur'].unique())
    
    selected_user = st.selectbox(
        "Select a user ID:",
        all_users,
        index=0
    )
    
    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        key="user_rec_slider"
    )
    
    # Show user's borrowing history
    user_books = st.session_state.data_loader.get_user_borrowed_books(selected_user)
    user_category = st.session_state.data_loader.get_user_category(selected_user)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Books Borrowed", len(user_books))
    with col2:
        st.metric("User Category", user_category)
    
    with st.expander("📚 View User's Borrowing History"):
        st.write(user_books)
    
    if st.button("Get Personalized Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            recommendations = st.session_state.recommendation_engine.get_recommendations_for_user(
                selected_user,
                num_recommendations=num_recommendations
            )
            
            if recommendations is not None and len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} personalized recommendations!")
                
                # Display recommendations
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{idx + 1}. {row['Titre']}**")
                            if 'Auteur_merged1' in row and pd.notna(row['Auteur_merged1']):
                                st.markdown(f"*Author: {row['Auteur_merged1']}*")
                            if 'topic_fr' in row and pd.notna(row['topic_fr']):
                                st.markdown(f"📑 Category: {row['topic_fr']}")
                        
                        with col2:
                            st.metric("Confidence", f"{row['confidence']:.3f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this user.")

elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Books",
        "👥 Users",
        "✍️ Authors",
        "🔗 Correlations"
    ])
    
    with tab1:
        st.markdown("### Book Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Borrowed Books")
            st.session_state.visualizations.show_top_books_chart(15)
        
        with col2:
            st.markdown("#### Category Distribution")
            st.session_state.visualizations.show_category_distribution()
    
    with tab2:
        st.markdown("### User Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Books per User Distribution")
            st.session_state.visualizations.show_books_per_user_distribution()
        
        with col2:
            st.markdown("#### User Categories")
            st.session_state.visualizations.show_user_category_distribution()
    
    with tab3:
        st.markdown("### Author Analytics")
        st.session_state.visualizations.show_top_authors(15)
        
        # Show author statistics
        author_stats = st.session_state.data_loader.get_author_statistics(20)
        if not author_stats.empty:
            st.markdown("#### Top 20 Authors")
            st.dataframe(author_stats, use_container_width=True, height=400)
    
    with tab4:
        st.markdown("### Book Correlations")
        st.markdown("""
        This heatmap shows which books are frequently borrowed together by the same users.
        Higher correlation (red) indicates books that are often borrowed together.
        """)
        
        top_n = st.slider("Number of books to include:", 10, 30, 20)
        st.session_state.visualizations.show_correlation_heatmap(top_n)

elif page == "📈 Association Rules":
    st.markdown('<div class="main-header">📈 Association Rules Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore the association rules discovered by the FP-Growth algorithm. These rules show 
    which books are frequently borrowed together and the strength of these associations.
    """)
    
    # Show rule statistics
    st.session_state.visualizations.show_association_rules_metrics(
        st.session_state.recommendation_engine
    )
    
    st.markdown("---")
    
    # Show top rules
    st.session_state.visualizations.show_top_association_rules(
        st.session_state.recommendation_engine,
        top_n=20
    )
    
    # Show detailed rule statistics
    st.markdown("---")
    st.markdown("### 📊 Rule Quality Metrics Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Support**: How frequently the itemset appears in the dataset
        - Higher support = more common pattern
        - Range: 0 to 1
        """)
        
        st.markdown("""
        **Confidence**: How often the rule is true
        - Probability of consequent given antecedent
        - Range: 0 to 1
        """)
    
    with col2:
        st.markdown("""
        **Lift**: How much more likely the consequent is with the antecedent
        - Lift > 1: Positive correlation
        - Lift = 1: No correlation
        - Lift < 1: Negative correlation
        """)
        
        st.markdown("""
        **Conviction**: Measure of implication strength
        - Higher conviction = stronger rule
        - Range: 0 to infinity
        """)

elif page == "👥 User Segmentation":
    st.markdown('<div class="main-header">👥 User Segmentation Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Analyze user behavior and segmentation based on borrowing patterns.
    Users are categorized into three groups based on their borrowing frequency.
    """)
    
    # Show segmentation pie chart
    st.session_state.visualizations.show_user_segmentation_pie()
    
    st.markdown("---")
    
    # Show borrowing by user type
    st.markdown("### 📊 Borrowing Patterns by User Type")
    st.session_state.visualizations.show_borrowing_by_student_year()
    
    st.markdown("---")
    
    # Show detailed segmentation statistics
    st.markdown("### 📈 Segmentation Statistics")
    
    if st.session_state.data_loader.user_stats is not None:
        segmentation_stats = st.session_state.data_loader.user_stats.groupby('category').agg({
            'total_borrowings': ['mean', 'median', 'min', 'max']
        }).round(2)
        
        st.dataframe(segmentation_stats, use_container_width=True)
        
        # Show distribution of borrowings within each category
        col1, col2, col3 = st.columns(3)
        
        categories = ['Light Reader', 'Moderate Reader', 'Heavy Reader']
        
        for col, category in zip([col1, col2, col3], categories):
            with col:
                st.markdown(f"#### {category}")
                category_data = st.session_state.data_loader.user_stats[
                    st.session_state.data_loader.user_stats['category'] == category
                ]
                
                if not category_data.empty:
                    st.metric("Users", len(category_data))
                    st.metric("Avg Borrowings", f"{category_data['total_borrowings'].mean():.1f}")
                    st.metric("Total Borrowings", f"{category_data['total_borrowings'].sum():,}")

elif page == "⚙️ Model Settings":
    st.markdown('<div class="main-header">⚙️ Model Settings</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Adjust the parameters of the recommendation model and retrain if needed.
    **Note**: Retraining the model may take a few minutes.
    """)
    
    # Current parameters
    st.markdown("### Current Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Minimum Support", st.session_state.recommendation_engine.min_support)
        st.markdown("""
        **Support** determines how frequently an itemset must appear to be considered.
        - Lower values: More rules, but potentially weaker patterns
        - Higher values: Fewer rules, but stronger patterns
        - Recommended: 0.003 - 0.01
        """)
    
    with col2:
        st.metric("Minimum Confidence", st.session_state.recommendation_engine.min_confidence)
        st.markdown("""
        **Confidence** determines the strength of the association rule.
        - Lower values: More recommendations, but potentially less relevant
        - Higher values: Fewer recommendations, but more relevant
        - Recommended: 0.1 - 0.5
        """)
    
    st.markdown("---")
    
    # Parameter adjustment
    st.markdown("### Adjust Parameters")
    
    new_support = st.slider(
        "Minimum Support:",
        min_value=0.001,
        max_value=0.02,
        value=st.session_state.recommendation_engine.min_support,
        step=0.001,
        format="%.4f"
    )
    
    new_confidence = st.slider(
        "Minimum Confidence:",
        min_value=0.05,
        max_value=0.8,
        value=st.session_state.recommendation_engine.min_confidence,
        step=0.05,
        format="%.2f"
    )
    
    if st.button("🔄 Retrain Model", type="primary"):
        with st.spinner("Retraining model... This may take a few minutes."):
            st.session_state.recommendation_engine.retrain_model(
                min_support=new_support,
                min_confidence=new_confidence
            )
            st.success("✅ Model retrained successfully!")
            st.rerun()
    
    st.markdown("---")
    
    # Model information
    st.markdown("### Model Information")
    
    stats = st.session_state.recommendation_engine.get_rule_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Itemsets", f"{stats['total_itemsets']:,}")
    
    with col2:
        st.metric("Total Rules", f"{stats['total_rules']:,}")
    
    with col3:
        st.metric("Strong Rules (Lift>1.5)", f"{stats['strong_rules']:,}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    📚 Library Book Recommendation System | Powered by FP-Growth & Association Rules
</div>
""", unsafe_allow_html=True)