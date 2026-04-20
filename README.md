# Netflix Movies and TV Shows Clustering 🎬🎥

![Netflix Insights](https://img.shields.io/badge/Netflix-Clustering_Project-E50914?style=for-the-badge&logo=netflix&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Algorithms-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Project Overview
This project focuses on building an end-to-end **unsupervised machine learning pipeline** to naturally cluster Netflix's vast catalog of Movies and TV Shows. By grouping together similar content, we can derive valuable insights into content distribution, characteristics, genres, and potentially improve recommendation systems and content discovery.

## 🎯 Objective
- **Exploratory Data Analysis (EDA):** Gain a deep understanding of the Netflix catalog, identifying distributions of categories, countries, cast members, and release trends.
- **Natural Language Processing (NLP):** Preprocess text-based metadata to extract meaningful features for clustering.
- **Clustering / Unsupervised Learning:** Apply diverse unsupervised clustering algorithms to segregate the dataset into logical content groupings based on textual similarity and metadata similarities.

## ⚙️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **NLP & Feature Engineering:** `nltk`, `scikit-learn` (TF-IDF Vectorizer)
- **Machine Learning (Clustering):** K-Means, Agglomerative Hierarchical Clustering, DBSCAN
- **Dimensionality Reduction:** TruncatedSVD / PCA

## 🚀 Project Workflow

### 1. Data Wrangling & Exploratory Data Analysis (EDA)
- **Notebook:** `Sample_EDA_Submission_Template.ipynb`
- Scrubbed and cleaned the dataset to handle missing values and inconsistencies.
- Conducted exhaustive Univariate, Bivariate, and Multivariate analysis (over 15+ descriptive visualizations).
- Analyzed volume of content added over years, top producing countries, and distribution of content ratings.

### 2. Feature Engineering & Selection
- **Text Preprocessing:** Cleaned and tokenized textual data (descriptions, genres, etc.) and applied stemming/lemmatization.
- **Vectorization:** Implemented **TF-IDF Vectorizer** globally across combined text-features, converting them into a mathematical matrix.
- **Dimensionality Reduction:** Used **TruncatedSVD / PCA** to collapse thousands of high-dimensional text vectors into a few principal components while preserving maximum variance.

### 3. Machine Learning & Clustering Model Creation
- **Notebook:** `Sample_ML_Submission_Template.ipynb`
- Built and evaluated three unsupervised ML models:
  - **K-Means Clustering:** Computed optimal $K$ value using the Elbow curve and Silhouette Score.
  - **Agglomerative Clustering:** Represented hierarchical relationships by building dendrograms.
  - **DBSCAN:** Used for density-based clustering to handle noise and anomalies seamlessly.

## 📁 Repository Structure
```
├── NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv # Raw Dataset
├── Sample_EDA_Submission_Template.ipynb       # Detailed EDA Notebook
├── Sample_ML_Submission_Template.ipynb        # ML & NLP Clustering Notebook
├── kmeans_model.pkl                           # Saved K-Means clustering model
├── svd_model.pkl                              # Saved TruncatedSVD dimensionality reduction model
├── tfidf_vectorizer.pkl                       # Saved fitted TF-IDF Vectorizer
└── README.md                                  # You are here!
```

## 🛠️ How to Use
1. **Clone the Repo:**
   ```sh
   git clone https://github.com/Deadsunx/Netflix-project.git
   cd Netflix-project
   ```
2. **Install Dependencies:**
   Ensure you have Jupyter installed alongside standard data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`).
3. **Run Notebooks:**
   Fire up Jupyter Notebook or Jupyter Lab from your terminal:
   ```sh
   jupyter notebook
   ```
   Open `Sample_EDA_Submission_Template.ipynb` first for insights, followed by `Sample_ML_Submission_Template.ipynb` to view the clustering mechanism.

## 📈 Future Scope
- Develop a content-based recommendation engine utilizing the generated clusters and Cosine Similarity scores.
- Build a Python/Streamlit web application for users to input a movie name and get real-time similar movie suggestions.
