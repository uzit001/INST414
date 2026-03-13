# Movie Similarity Analysis Using MovieLens

This project explores how similarity metrics can be used to identify movies that are most similar to a given film. Using the MovieLens dataset, I computed similarity scores between movies based on their genres, average ratings, and release years.

The goal is to demonstrate how a simple similarity model can support a movie recommendation system similar to those used by streaming platforms.

---

## Dataset

This analysis uses the MovieLens dataset created by GroupLens Research at the University of Minnesota.

Dataset used:
- **ml-latest-small**

Dataset contents:
- ~9,700 movies
- ~100,000 user ratings
- Genre metadata for each movie

Files used:
- `movies.csv`
- `ratings.csv`

Dataset source:  
https://grouplens.org/datasets/movielens/latest/

---

## Project Objective

The objective of this project is to answer the following question:

**If a viewer enjoys a particular movie, what other movies are most similar and should be recommended next?**

Three query movies were analyzed:

- The Matrix (1999)
- Inception (2010)
- Interstellar (2014)

For each movie, the analysis identifies the **10 most similar movies** in the dataset.

---

## Features Used

Each movie was represented using the following features:

1. **Genres**  
   Genres were converted into numerical features using one-hot encoding.

2. **Average Rating**  
   The average user rating for each movie was calculated from the ratings dataset.

3. **Release Year**  
   The release year was extracted from the movie title.

These features were combined into a feature vector representing each movie.

---

## Similarity Method

Similarity between movies was calculated using **cosine similarity**.

Cosine similarity measures the angle between two feature vectors. Movies with similar genres, ratings, and release years will have higher similarity scores.

Before computing similarity, features were standardized to ensure that rating and year values did not dominate the genre features.

---

## Tools and Libraries

The analysis was performed using Python and the following libraries:

- pandas
- scikit-learn
- matplotlib
- seaborn

These tools were used for data processing, similarity calculations, and visualizations.

---

## Project Structure

Example repository structure:

```
movie-similarity-analysis/
│
├── movie_similarity.ipynb
├── README.md
├── movies.csv
├── ratings.csv
```

---

## Results

The model generated similarity rankings for each query movie and visualizations including:

- Top-10 most similar movies for each query
- Similarity bar charts
- A similarity heatmap comparing the query movies

These results illustrate how similarity analysis can support movie recommendation systems.

---

## Limitations

This analysis has several limitations:

- Only genres, ratings, and release year were used as features
- Plot summaries, actors, and directors were not included
- Ratings may reflect biases from MovieLens users

More advanced recommendation systems typically incorporate additional metadata and collaborative filtering techniques.

---

## How to Run the Project

1. Download the MovieLens dataset from the GroupLens website.
2. Place `movies.csv` and `ratings.csv` in the project directory.
3. Open the Jupyter notebook:

```
movie_similarity.ipynb
```

4. Run the notebook to reproduce the analysis and visualizations.

---

## Medium Article

A full explanation of this project is provided in the accompanying Medium article:

https://medium.com/@uzzamtariq/finding-movies-similar-to-the-matrix-inception-interstellar-using-cosine-similarity-fdc9666a185d

