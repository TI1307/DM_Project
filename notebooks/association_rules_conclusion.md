# Advanced Association Rules Analysis — Summary

Through FP-Growth data mining on library borrowing records, we discovered meaningful relationships between books, enabling evidence-based recommendations.

## 1. Optimal Rule Generation
- **Minimum Support:** 0.005 (0.5%) → captures significant patterns without noise from rare borrowings.  
- **Minimum Confidence:** 0.3 (30%) → ensures recommendations are relevant and reliable.  
- **Result:** Balanced coverage and strong predictive rules.

## 2. Discovering Strong Associations
- **Top 30 Rules:** Exceptional predictive power.  
- **Average Lift:** 22.17 → borrowing one book increases likelihood of borrowing associated book 20+ times versus random.  
- **Strategic Clusters:**  
  - Academic pathways (e.g., Computer Vision → Neural Networks → Deep Learning, Lift: 135.5)  
  - Thematic links in mathematics and algebra.  

## 3. Actionable Library Insights
- **Thematic Cross-Promotion:** Users borrow "knowledge paths," not just individual books.  
- **Quality over Density:** Even users borrowing 1-2 books can receive accurate recommendations based on top rules.  

## 4. Final Summary
- **FP-Growth:** Efficient and scalable for this dataset.  
- **Application:** Provides scientific basis for a recommendation system ("Users who liked this also liked…") grounded in actual borrowing behavior.
