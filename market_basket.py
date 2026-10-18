

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Sample shopping transactions
transactions = [
    ["Bread", "Butter", "Milk"],
    ["Bread", "Butter"],
    ["Bread", "Milk"],
    ["Butter", "Milk"],
    ["Bread", "Butter", "Milk"],
    ["Bread", "Butter"],
    ["Milk", "Butter"],
    ["Bread", "Milk", "Butter"],
]

# Step 2: Convert transactions into a DataFrame
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded_data, columns=encoder.columns_)

print("\n--- Encoded Transaction Data ---")
print(df)

# Step 3: Find frequent itemsets
frequent_itemsets = apriori(
    df,
    min_support=0.25,
    use_colnames=True
)

print("\n--- Frequent Itemsets ---")
print(frequent_itemsets)

# Step 4: Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("\n--- Association Rules ---")
if not rules.empty:
    print(rules[["antecedents", "consequents",
                 "support", "confidence", "lift"]])
else:
    print("No association rules found.")

import matplotlib.pyplot as plt

# Graph 1: Top frequent itemsets
if not frequent_itemsets.empty:
    plot_data = frequent_itemsets.copy()

    plot_data["Items"] = plot_data["itemsets"].apply(
        lambda items: ", ".join(sorted(items))
    )

    plot_data = plot_data.sort_values(
        by="support", ascending=False
    ).head(10)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_data["Items"], plot_data["support"])
    plt.title("Top Frequent Itemsets")
    plt.xlabel("Product Combinations")
    plt.ylabel("Support")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("frequent_itemsets.png")
    plt.show()

# Graph 2: Association rules
if not rules.empty:
    rules_plot = rules.copy()

    rules_plot["Rule"] = rules_plot.apply(
        lambda row:
        ", ".join(sorted(row["antecedents"]))
        + " -> "
        + ", ".join(sorted(row["consequents"])),
        axis=1
    )

    rules_plot = rules_plot.sort_values(
        by="confidence", ascending=False
    ).head(10)

    plt.figure(figsize=(9, 5))
    plt.bar(rules_plot["Rule"], rules_plot["confidence"])
    plt.title("Top Association Rules by Confidence")
    plt.xlabel("Association Rules")
    plt.ylabel("Confidence")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("association_rules.png")
    plt.show()

print("\nGraphs saved successfully!")
