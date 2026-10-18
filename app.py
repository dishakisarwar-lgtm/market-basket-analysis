
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("Market Basket Analysis")
st.write("Discover frequently purchased product combinations.")

try:
    data = pd.read_csv("transactions.csv")

    transactions = data["Products"].dropna().apply(
        lambda x: [item.strip() for item in x.split(",")]
    ).tolist()

    st.subheader("Transaction Dataset")
    st.dataframe(data, use_container_width=True)

    min_support = st.slider(
        "Minimum Support", 0.1, 0.9, 0.2, 0.1
    )
    min_confidence = st.slider(
        "Minimum Confidence", 0.1, 1.0, 0.5, 0.1
    )

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(encoded, columns=encoder.columns_)

    itemsets = apriori(
        df, min_support=min_support, use_colnames=True
    )

    st.subheader("Frequent Product Combinations")

    if not itemsets.empty:
        display_items = itemsets.copy()
        display_items["Products"] = display_items["itemsets"].apply(
            lambda x: ", ".join(sorted(x))
        )

        st.dataframe(
            display_items[["Products", "support"]],
            use_container_width=True
        )

        chart_data = display_items.nlargest(10, "support")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(chart_data["Products"], chart_data["support"])
        ax.set_ylabel("Support")
        ax.set_title("Top Frequent Product Combinations")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if len(itemsets) >= 2:
            rules = association_rules(
                itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            st.subheader("Product Association Rules")

            if not rules.empty:
                rules["If Customer Buys"] = rules["antecedents"].apply(
                    lambda x: ", ".join(sorted(x))
                )
                rules["Recommend"] = rules["consequents"].apply(
                    lambda x: ", ".join(sorted(x))
                )

                result = rules[
                    ["If Customer Buys", "Recommend",
                     "support", "confidence", "lift"]
                ].round(3)

                st.dataframe(result, use_container_width=True)

                st.download_button(
                    "Download Association Rules CSV",
                    data=result.to_csv(index=False),
                    file_name="association_rules.csv",
                    mime="text/csv"
                )
            else:
                st.info("No rules found. Lower the confidence threshold.")
        else:
            st.info("Not enough frequent itemsets to generate rules.")

    else:
        st.info("No frequent itemsets found. Lower minimum support.")

except FileNotFoundError:
    st.error("transactions.csv is missing from the project folder.")

except Exception as e:
    st.error(f"An error occurred: {e}")
