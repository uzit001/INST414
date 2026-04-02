Startup Company Clustering Analysis
INST414 – Data Science Techniques | Assignment 4
A K-Means clustering analysis of 30,993 funded startups to uncover distinct company archetypes and their relationship to acquisition outcomes.

📌 Project Overview
Research Question: What types of startups exist based on their funding profile, and which archetype is most likely to get acquired?
Stakeholder: Early-stage VC analysts looking for a data-driven framework to triage investment opportunities.
Approach: K-Means clustering (k=4) on three features — total funding, number of funding rounds, and company age — using Euclidean distance on normalized features.

📁 Repository Structure
Startup-companies-clustering/
│
├── clustering.ipynb          # Main analysis notebook
├── investments_PartI.csv     # Raw dataset (download from Kaggle - see below)
├── elbow_plot.png            # Elbow method figure
├── cluster_plot.png          # Cluster visualization figures
└── README.md                 # This file

📦 Dataset
Source: Crunchbase Startup Investments Dataset on Kaggle
The dataset is not included in this repository due to file size. To reproduce this analysis:

Download investments_PartI.csv from the Kaggle link above (free account required)
Place the CSV file in the root of this project folder alongside clustering.ipynb

Raw dataset: 54,294 companies × 39 fields
After cleaning: 30,993 companies

🔧 Setup & Installation
Requirements

Python 3.8+
Jupyter Notebook or VS Code with Jupyter extension

Install dependencies
bashpip install pandas numpy scikit-learn matplotlib seaborn
Run the notebook
Open clustering.ipynb in Jupyter or VS Code and run all cells in order.

🧹 Data Cleaning Notes
A few issues you'll encounter with the raw data (already handled in the notebook):
BugFixColumn names have hidden whitespace (e.g., ' market ')df.columns = df.columns.str.strip()funding_total_usd stored as string with commas ("99,99,999").str.replace(',', '').str.strip() before pd.to_numeric()Implausible founding years (e.g., 1902)Filter to founded_year >= 1990Extreme funding outliersFilter to $10,000 ≤ funding ≤ $1,000,000,000

📊 Methods
StepChoiceJustificationFeaturesfunding_total_usd, funding_rounds, company_ageNumeric, directly measurable, relevant to investment decisionsPreprocessingStandardScaler normalizationPrevents high-magnitude funding values from dominating distanceSimilarity metricEuclidean distanceStandard for K-Means; appropriate for normalized continuous featuresClustering algorithmK-Means (sklearn.cluster.KMeans)Efficient, interpretable, well-suited for numeric dataK selectionElbow method (k=2 to k=10)Elbow visually apparent at k=4; diminishing returns beyond thatValidationSilhouette scoreScore of 0.51 confirms strong cluster separation

🔍 Results: The Four Startup Archetypes
ClusterLabelAvg FundingAvg RoundsAvg AgeSizeAcquisition Rate0🌱 Early-stage Seedlings$3.9M1.54.5 yrs20,6945.0%1🕰️ Mature Bootstrappers$14.6M1.614.3 yrs6,17517.3%2🚀 Mega-funded Giants$353M4.510.5 yrs2769.9%3💼 Mid-stage Growth$40M4.98.3 yrs3,84711.8%
Key finding: Acquisition rate is not driven by funding size alone. Mature, capital-efficient companies (Cluster 1) and multi-round mid-stage companies (Cluster 3) outperform even mega-funded giants in acquisition likelihood.

📚 Libraries Used

pandas — data loading and cleaning
numpy — numerical operations
scikit-learn — StandardScaler, KMeans, silhouette_score
matplotlib — plotting
seaborn — visualization styling


⚠️ Limitations

Survivorship bias: Crunchbase skews toward US companies and active profile maintainers
Static snapshot: Data reflects ~2015; cluster membership shifts as companies evolve
Three features only: Revenue, team size, and market category are not included
Correlation ≠ causation: Acquisition rate differences describe patterns, not mechanisms


📝 Medium Post
Read the full write-up and analysis here:
🔗 (Add your Medium post URL here after publishing)

👤 Author
uzit001 | INST414 – Data Science Techniques
