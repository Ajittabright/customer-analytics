# =============================================================================
# src/eda.py
# =============================================================================
# PURPOSE:
#   Exploratory Data Analysis — understand the data through charts.
#   Every chart answers a specific business question.
#   All charts are saved to reports/figures/ folder.
#
# CHARTS GENERATED:
#   1. Churn Distribution (Bar Chart)
#   2. Age Distribution (Histogram)
#   3. Churn by Geography (Bar Chart)
#   4. Churn by Gender (Bar Chart)
#   5. Credit Score Distribution (Histogram)
#   6. Balance Distribution (Histogram)
#   7. Correlation Heatmap
#   8. Age vs Balance Scatter Plot
#   9. Numerical Features Box Plot
#  10. Churn Rate by Age Group
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt        # Main plotting library
import matplotlib.patches as mpatches  # For custom legend items
import seaborn as sns                  # Statistical visualization (built on matplotlib)


# -----------------------------------------------------------------------------
# LOGGER
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FUNCTION 1: setup_plot_style()
# PURPOSE: Set consistent visual style for ALL charts
#
# WHY?
# Professional reports have consistent styling.
# We set this ONCE and all charts automatically follow it.
# -----------------------------------------------------------------------------
def setup_plot_style(config: dict) -> None:
    """
    Set up matplotlib and seaborn plot styling.

    Args:
        config (dict): Configuration dictionary

    Returns:
        None
    """

    # Get style settings from config
    plot_style = config["eda"]["plot_style"]
    color_palette = config["eda"]["color_palette"]

    # -----------------------------------------------------------------
    # plt.style.use() sets the overall look of all charts
    # "seaborn-v0_8" gives clean, professional looking charts
    # -----------------------------------------------------------------
    try:
        plt.style.use(plot_style)
    except Exception:
        # If style not found, use default
        plt.style.use("default")
        logger.warning(f"Style '{plot_style}' not found, using default")

    # Set color palette for seaborn charts
    sns.set_palette(color_palette)

    # Set font sizes globally
    plt.rcParams.update({
        "font.size": 12,           # Default font size
        "axes.titlesize": 14,      # Chart title size
        "axes.labelsize": 12,      # Axis label size
        "xtick.labelsize": 10,     # X-axis tick label size
        "ytick.labelsize": 10,     # Y-axis tick label size
        "figure.titlesize": 16     # Figure title size
    })

    logger.info("Plot style configured successfully")


# -----------------------------------------------------------------------------
# FUNCTION 2: save_plot()
# PURPOSE: Save a matplotlib figure to the reports/figures/ folder
#
# WHY A SEPARATE FUNCTION?
# Every chart needs to be saved. Instead of repeating save code
# in every chart function, we have ONE save function.
# DRY Principle: Don't Repeat Yourself!
# -----------------------------------------------------------------------------
def save_plot(fig: plt.Figure, filename: str, config: dict) -> None:
    """
    Save a matplotlib figure to the reports directory.

    Args:
        fig (plt.Figure): The matplotlib figure to save
        filename (str): Name of the file (e.g., "churn_distribution.png")
        config (dict): Configuration dictionary

    Returns:
        None
    """

    # Get the reports directory from config
    reports_dir = config["paths"]["reports_dir"]

    # Create directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)

    # Full file path
    filepath = os.path.join(reports_dir, filename)

    # -----------------------------------------------------------------
    # savefig() saves the figure to disk
    # dpi = dots per inch (higher = sharper image)
    # bbox_inches="tight" removes extra whitespace around the chart
    # -----------------------------------------------------------------
    fig.savefig(
        filepath,
        dpi=config["eda"]["dpi"],
        bbox_inches="tight",
        facecolor="white"    # White background
    )

    logger.info(f"  Chart saved: {filepath}")

    # Close the figure to free memory
    # Without this, matplotlib keeps all figures in memory!
    plt.close(fig)


# -----------------------------------------------------------------------------
# FUNCTION 3: plot_churn_distribution()
# PURPOSE: Show how many customers churned vs stayed
#
# BUSINESS QUESTION: "How balanced is our dataset?"
# This tells us if we have an imbalanced dataset problem.
# -----------------------------------------------------------------------------
def plot_churn_distribution(df: pd.DataFrame, config: dict) -> None:
    """
    Plot the distribution of churned vs non-churned customers.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 1: Churn Distribution...")

    # Get figure size from config
    fig_size = tuple(config["eda"]["figure_size"])

    # -----------------------------------------------------------------
    # Create figure and axes
    # fig = the overall figure (like a canvas)
    # axes = the actual plot area (like the drawing on the canvas)
    # 1, 2 means: 1 row, 2 columns of subplots
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=fig_size)

    # Get target column name from config
    target = config["data"]["target_column"]

    # Count values: {0: 7963, 1: 2037}
    churn_counts = df[target].value_counts()
    churn_labels = ["Stayed (0)", "Churned (1)"]
    colors = ["#2ecc71", "#e74c3c"]  # Green for stayed, Red for churned

    # -----------------------------------------------------------------
    # LEFT PLOT: Bar Chart
    # axes[0] = first subplot (left)
    # -----------------------------------------------------------------
    axes[0].bar(
        churn_labels,
        churn_counts.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )
    axes[0].set_title("Churn Count", fontweight="bold")
    axes[0].set_xlabel("Customer Status")
    axes[0].set_ylabel("Number of Customers")

    # Add count labels on top of each bar
    for i, count in enumerate(churn_counts.values):
        axes[0].text(
            i,              # x position
            count + 50,     # y position (slightly above bar)
            f"{count:,}",   # text to display (with comma formatting)
            ha="center",    # horizontal alignment
            fontweight="bold"
        )

    # -----------------------------------------------------------------
    # RIGHT PLOT: Pie Chart
    # axes[1] = second subplot (right)
    # -----------------------------------------------------------------
    axes[1].pie(
        churn_counts.values,
        labels=churn_labels,
        colors=colors,
        autopct="%1.1f%%",    # Show percentage with 1 decimal
        startangle=90,         # Start from top
        explode=(0, 0.05)      # Slightly separate the churned slice
    )
    axes[1].set_title("Churn Percentage", fontweight="bold")

    # Overall figure title
    fig.suptitle(
        "Customer Churn Distribution",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the chart
    save_plot(fig, "01_churn_distribution.png", config)
    logger.info("  Chart 1 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 4: plot_age_distribution()
# PURPOSE: Understand the age spread of customers
#
# BUSINESS QUESTION: "What age group is most likely to churn?"
# -----------------------------------------------------------------------------
def plot_age_distribution(df: pd.DataFrame, config: dict) -> None:
    """
    Plot age distribution split by churn status.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 2: Age Distribution...")

    fig_size = tuple(config["eda"]["figure_size"])
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    target = config["data"]["target_column"]

    # -----------------------------------------------------------------
    # LEFT: Histogram of all ages
    # bins=30 means divide age range into 30 equal buckets
    # edgecolor="black" adds border to each bar
    # -----------------------------------------------------------------
    axes[0].hist(
        df["Age"],
        bins=30,
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.7          # alpha = transparency (0=invisible, 1=solid)
    )
    axes[0].set_title("Overall Age Distribution", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Number of Customers")

    # Add vertical line for mean age
    mean_age = df["Age"].mean()
    axes[0].axvline(
        mean_age,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean Age: {mean_age:.1f}"
    )
    axes[0].legend()

    # -----------------------------------------------------------------
    # RIGHT: Age distribution split by churn
    # We overlay two histograms — one for churned, one for stayed
    # -----------------------------------------------------------------
    stayed = df[df[target] == 0]["Age"]
    churned = df[df[target] == 1]["Age"]

    axes[1].hist(stayed, bins=30, alpha=0.6, color="#2ecc71", label="Stayed", edgecolor="black", linewidth=0.3)
    axes[1].hist(churned, bins=30, alpha=0.6, color="#e74c3c", label="Churned", edgecolor="black", linewidth=0.3)
    axes[1].set_title("Age Distribution by Churn Status", fontweight="bold")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Number of Customers")
    axes[1].legend()

    fig.suptitle("Customer Age Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "02_age_distribution.png", config)
    logger.info("  Chart 2 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 5: plot_geography_analysis()
# PURPOSE: See which countries have highest churn rates
#
# BUSINESS QUESTION: "Should we focus retention efforts on specific regions?"
# -----------------------------------------------------------------------------
def plot_geography_analysis(df: pd.DataFrame, config: dict) -> None:
    """
    Plot churn analysis by geography.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 3: Geography Analysis...")

    # Check if Geography column exists (before encoding)
    # If already encoded, skip this chart
    if "Geography" not in df.columns:
        logger.warning("  Geography column not found (already encoded). Skipping.")
        return

    fig_size = tuple(config["eda"]["figure_size"])
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    target = config["data"]["target_column"]

    # Count customers per geography
    geo_counts = df["Geography"].value_counts()

    # LEFT: Customer count per geography
    axes[0].bar(
        geo_counts.index,
        geo_counts.values,
        color=["#3498db", "#e74c3c", "#2ecc71"],
        edgecolor="black",
        linewidth=0.5
    )
    axes[0].set_title("Customers by Geography", fontweight="bold")
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Number of Customers")

    for i, count in enumerate(geo_counts.values):
        axes[0].text(i, count + 20, f"{count:,}", ha="center", fontweight="bold")

    # RIGHT: Churn rate per geography
    # groupby() groups rows by Geography
    # then we calculate mean of Exited (mean of 0s and 1s = churn rate!)
    churn_by_geo = df.groupby("Geography")[target].mean() * 100

    axes[1].bar(
        churn_by_geo.index,
        churn_by_geo.values,
        color=["#3498db", "#e74c3c", "#2ecc71"],
        edgecolor="black",
        linewidth=0.5
    )
    axes[1].set_title("Churn Rate by Geography (%)", fontweight="bold")
    axes[1].set_xlabel("Country")
    axes[1].set_ylabel("Churn Rate (%)")

    for i, rate in enumerate(churn_by_geo.values):
        axes[1].text(i, rate + 0.5, f"{rate:.1f}%", ha="center", fontweight="bold")

    fig.suptitle("Geography Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "03_geography_analysis.png", config)
    logger.info("  Chart 3 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 6: plot_correlation_heatmap()
# PURPOSE: Show relationships between ALL numerical features
#
# BUSINESS QUESTION: "Which features are most related to churn?"
#
# WHAT IS CORRELATION?
# Correlation measures how two variables move together.
# Range: -1 to +1
# +1 = perfectly positive (one goes up, other goes up)
# -1 = perfectly negative (one goes up, other goes down)
#  0 = no relationship
# -----------------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, config: dict) -> None:
    """
    Plot correlation heatmap of all numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 4: Correlation Heatmap...")

    # Select only numerical columns for correlation
    numerical_df = df.select_dtypes(include=[np.number])

    # Calculate correlation matrix
    # .corr() calculates pairwise correlation between all columns
    corr_matrix = numerical_df.corr()

    # Make figure larger for heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # -----------------------------------------------------------------
    # sns.heatmap() draws the correlation matrix as colored grid
    # annot=True shows the correlation values in each cell
    # fmt=".2f" formats numbers to 2 decimal places
    # cmap="RdYlGn" = Red(negative) Yellow(zero) Green(positive)
    # center=0 means 0 correlation = yellow (middle color)
    # -----------------------------------------------------------------
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 8}
    )

    ax.set_title(
        "Feature Correlation Heatmap\n(Green=Positive, Red=Negative)",
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()
    save_plot(fig, "04_correlation_heatmap.png", config)
    logger.info("  Chart 4 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 7: plot_balance_analysis()
# PURPOSE: Understand account balance patterns
#
# BUSINESS QUESTION: "Do customers with zero balance churn more?"
# -----------------------------------------------------------------------------
def plot_balance_analysis(df: pd.DataFrame, config: dict) -> None:
    """
    Plot balance distribution and its relationship with churn.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 5: Balance Analysis...")

    fig_size = tuple(config["eda"]["figure_size"])
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    target = config["data"]["target_column"]

    # LEFT: Balance distribution
    axes[0].hist(
        df["Balance"],
        bins=50,
        color="#9b59b6",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.7
    )
    axes[0].set_title("Account Balance Distribution", fontweight="bold")
    axes[0].set_xlabel("Balance")
    axes[0].set_ylabel("Number of Customers")

    # Add mean line
    mean_balance = df["Balance"].mean()
    axes[0].axvline(
        mean_balance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: ${mean_balance:,.0f}"
    )
    axes[0].legend()

    # RIGHT: Balance by churn status (Box Plot)
    # Box plot shows: min, Q1, median, Q3, max, and outliers
    stayed_balance = df[df[target] == 0]["Balance"]
    churned_balance = df[df[target] == 1]["Balance"]

    axes[1].boxplot(
        [stayed_balance, churned_balance],
        tick_labels=["Stayed", "Churned"],
        patch_artist=True,       # Fill boxes with color
        boxprops=dict(facecolor="#3498db", alpha=0.7),
        medianprops=dict(color="red", linewidth=2)
    )
    axes[1].set_title("Balance by Churn Status", fontweight="bold")
    axes[1].set_xlabel("Customer Status")
    axes[1].set_ylabel("Account Balance")

    fig.suptitle("Account Balance Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "05_balance_analysis.png", config)
    logger.info("  Chart 5 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 8: plot_credit_score_analysis()
# PURPOSE: Analyze credit score patterns
#
# BUSINESS QUESTION: "Do low credit score customers churn more?"
# -----------------------------------------------------------------------------
def plot_credit_score_analysis(df: pd.DataFrame, config: dict) -> None:
    """
    Plot credit score distribution and churn relationship.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 6: Credit Score Analysis...")

    fig_size = tuple(config["eda"]["figure_size"])
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    target = config["data"]["target_column"]

    # LEFT: Credit score distribution
    axes[0].hist(
        df["CreditScore"],
        bins=40,
        color="#e67e22",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.7
    )
    axes[0].set_title("Credit Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Credit Score")
    axes[0].set_ylabel("Number of Customers")

    # RIGHT: Credit score by churn (violin plot)
    # Violin plot = box plot + density distribution
    # Shows WHERE most values are concentrated
    stayed_credit = df[df[target] == 0]["CreditScore"]
    churned_credit = df[df[target] == 1]["CreditScore"]

    axes[1].violinplot(
        [stayed_credit, churned_credit],
        positions=[1, 2],
        showmeans=True,
        showmedians=True
    )
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Stayed", "Churned"])
    axes[1].set_title("Credit Score by Churn Status", fontweight="bold")
    axes[1].set_xlabel("Customer Status")
    axes[1].set_ylabel("Credit Score")

    fig.suptitle("Credit Score Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "06_credit_score_analysis.png", config)
    logger.info("  Chart 6 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 9: plot_products_analysis()
# PURPOSE: See how number of products affects churn
#
# BUSINESS QUESTION: "Do customers with more products stay longer?"
# -----------------------------------------------------------------------------
def plot_products_analysis(df: pd.DataFrame, config: dict) -> None:
    """
    Plot number of products vs churn analysis.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dictionary
    """

    logger.info("Generating Chart 7: Products Analysis...")

    fig_size = tuple(config["eda"]["figure_size"])
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    target = config["data"]["target_column"]

    # Products distribution
    product_counts = df["NumOfProducts"].value_counts().sort_index()

    axes[0].bar(
        product_counts.index,
        product_counts.values,
        color="#1abc9c",
        edgecolor="black",
        linewidth=0.5
    )
    axes[0].set_title("Number of Products Distribution", fontweight="bold")
    axes[0].set_xlabel("Number of Products")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_xticks(product_counts.index)

    # Churn rate by number of products
    churn_by_products = df.groupby("NumOfProducts")[target].mean() * 100

    axes[1].bar(
        churn_by_products.index,
        churn_by_products.values,
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5
    )
    axes[1].set_title("Churn Rate by Number of Products", fontweight="bold")
    axes[1].set_xlabel("Number of Products")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_xticks(churn_by_products.index)

    for i, (prod, rate) in enumerate(churn_by_products.items()):
        axes[1].text(prod, rate + 0.5, f"{rate:.1f}%", ha="center", fontweight="bold")

    fig.suptitle("Number of Products Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "07_products_analysis.png", config)
    logger.info("  Chart 7 complete!")


# -----------------------------------------------------------------------------
# FUNCTION 10: run_eda()
# PURPOSE: Master function — runs ALL EDA charts in order
# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, config: dict) -> None:
    """
    Run complete Exploratory Data Analysis pipeline.

    Args:
        df (pd.DataFrame): Input DataFrame (use RAW/CLEANED data, not encoded)
        config (dict): Configuration dictionary

    Returns:
        None: All charts saved to reports/figures/
    """

    logger.info("=" * 60)
    logger.info("STARTING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    # Setup visual style first
    setup_plot_style(config)

    # Run all charts
    plot_churn_distribution(df, config)
    plot_age_distribution(df, config)
    plot_geography_analysis(df, config)
    plot_correlation_heatmap(df, config)
    plot_balance_analysis(df, config)
    plot_credit_score_analysis(df, config)
    plot_products_analysis(df, config)

    logger.info("=" * 60)
    logger.info("EDA COMPLETE!")
    logger.info(f"All charts saved to: {config['paths']['reports_dir']}")
    logger.info("=" * 60)


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import load_config, setup_logging, load_data
    from src.data_cleaner import clean_data

    # Step 1: Load config
    config = load_config()

    # Step 2: Setup logging
    setup_logging(config)

    # Step 3: Load raw data
    df_raw = load_data(config)

    # Step 4: Clean data
    # IMPORTANT: We use CLEANED data for EDA, NOT encoded data
    # Because Geography/Gender as text makes charts more readable!
    df_cleaned = clean_data(df_raw, config)

    # Step 5: Run EDA
    run_eda(df_cleaned, config)

    logger.info("\neda.py completed successfully!")