# %%
data = [
    ['milk', 'bread', 'butter'],
    ['beer', 'diapers', 'milk'],
    ['milk', 'diapers', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'diapers', 'bread', 'butter', 'beer']
]


# %%
from apyori import apriori

results = apriori(data,min_support=0.3,min_confidence=0.7)

# Display only frequent itemsets and association rules
for result in results:
    print("Frequent Itemset:", list(result.items))
    for rule in result.ordered_statistics:
        print("Rule:", list(rule.items_base), "->", list(rule.items_add))

# %%
# The Apriori algorithm is a popular algorithm used in association rule learning to find frequent itemsets and generate strong association rules based on specified thresholds of support and confidence. 
# The algorithm is especially useful for identifying relationships between items in large datasets, such as market basket analysis.

# %%
# Explanation:

#     Frequent Itemset: Displays the set of items that frequently appear together (above the minimum support threshold).
#     Association Rule: Shows the relationship between the items (antecedent → consequent) where confidence is above the threshold.


