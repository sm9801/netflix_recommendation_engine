# Netflix Recommendation Engine

This project is a machine learning-powered recommendation system designed to suggest movies, shows, and other programs to users based on their previous viewing preferences and ratings on Netflix. Using a combination of collaborative filtering techniques and clustering, the system predicts user preferences and tailors recommendations accordingly.

Full results and analysis can be found here: [Netflix Analysis](https://github.com/sm9801/Netflix_Recommendation_Engine/blob/master/Results%20copy.ipynb)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
  - [Collaborative Filtering with Gaussian Mixture Model (GMM)](#collaborative-filtering-with-gaussian-mixture-model-gmm)
  - [Prediction with Expectation-Maximization (EM) Algorithm](#prediction-with-expectation-maximization-em-algorithm)
  - [K-Means Clustering](#k-means-clustering)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Future Improvements](#future-improvements)

## Project Overview
This Netflix recommendation system uses collaborative filtering to generate tailored suggestions by predicting ratings for shows that users haven’t yet watched. By clustering similar users and using probabilistic modeling, the system uncovers and aligns with user preferences, enhancing user engagement with personalized recommendations.

### Hypothesis
- We expect K-means and GMM to perform similarly given the current dataset, but as underlying distribution becomes more complex and as complications are introduced, GMM will generalize better than K-means and will be reflected via test results.
- To test our hypothesis, after EDA is performed and parameters are optimized, we will put our GMM and K-means model through 3 stress tests:
  - Stress Test 1: Missing data at varying rates [0.1, 0.3, 0.5, 0.7]
  - Stress Test 2: Noise introduction [baseline, moderate, strong, adversarial]
  - Stress Test 3: Missing data at 80% and strong noise introduction
- The results of Stress Test 3 will be used to compute percentage improvement of the optimal GMM. 

## Dataset
The dataset contains information about user ratings. It is a matrix of 1200 * 1200 user rating data and missing data is represented as 0.
<img width="1784" height="919" alt="image" src="https://github.com/user-attachments/assets/7706ab95-fcee-4660-9607-26a11053b259" />
<img width="1784" height="919" alt="image" src="https://github.com/user-attachments/assets/ba14f562-4e3b-430e-b13e-c3433002528c" />



## Approach

### EDA via t-SNE
To understand the underlying data, we will apply t-SNE plots to both K-means and GMM.

### Collaborative Filtering with Gaussian Mixture Model (GMM)
To capture patterns in user behavior and generate personalized recommendations, the project uses a Gaussian Mixture Model (GMM) for collaborative filtering. GMM’s probabilistic approach enables the system to model complex distributions of user ratings, which improves the accuracy of predicted ratings.

### Prediction with Expectation-Maximization (EM) Algorithm
The Expectation-Maximization (EM) algorithm is used alongside GMM for optimizing predictions. EM iteratively refines cluster parameters to maximize the likelihood of the observed data, allowing the model to more precisely assign user preferences and enhance recommendation accuracy.

### Expectation-Maximization (EM) Algorithm

The Expectation-Maximization (EM) algorithm is an iterative method used to estimate parameters in models with latent variables, such as Gaussian Mixture Models (GMMs). In this recommendation system, the EM algorithm helps assign users to probabilistic clusters, capturing similarities in their preferences and improving rating predictions.

The EM algorithm alternates between two main steps until convergence:

1. **E-Step (Expectation Step)**: Calculate the expected value of the latent variables based on the observed data and the current parameter estimates.

   Given data  $X$ and current parameters $\theta$, compute the responsibilities (posterior probabilities) for each cluster $k$ :
   
   $\gamma_{z_i} = P(z_i | X; \theta) = \frac{\pi_k \, \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i | \mu_j, \Sigma_j)}$
   
   where:
   - $\gamma_{z_i}$ is the responsibility of cluster \( k \) for data point $x_i$,
   - $\pi_k$ is the prior probability of cluster $k$,
   - $\mathcal{N}(x_i | \mu_k, \Sigma_k)$ is the Gaussian distribution for cluster $k$ with mean $\mu_k$ and covariance $\Sigma_k$.

3. **M-Step (Maximization Step)**: Update the parameters $\theta = (\pi_k, \mu_k, \Sigma_k)$ by maximizing the expected log-likelihood from the E-step.

   The parameters are updated as follows:
   - **Update the mean**:
     
     $\mu_k = \frac{\sum_{i=1}^{N} \gamma_{z_i} \, x_i}{\sum_{i=1}^{N} \gamma_{z_i}}$
     
   - **Update the covariance**:
     
     $\Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{z_i} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{z_i}}$
     
   - **Update the mixture weights**:
     
     $\pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{z_i}$
     
   where $N$ is the total number of data points, and $K$ is the number of clusters.

These E and M steps repeat until convergence, yielding optimized parameters $\theta$ that enhance the model’s accuracy in predicting user ratings.


### K-Means Clustering
K-Means clustering was implemented to segment users into distinct groups based on their viewing preferences. Through experimentation, 3 was identified as the optimal number of clusters, with each cluster representing a unique subset of user tastes. This grouping helps further tailor recommendations by aligning users with others in the same cluster.

## Evaluation
The model’s performance was evaluated using the Root Mean Square Error (RMSE) metric, which compares the predicted ratings with actual user ratings. An RMSE score of 0.48 demonstrates the model’s ability to closely approximate true user preferences.

## Results
- **Optimal GMM Clusters**: 3
- **RMSE**: 0.48

The achieved RMSE score indicates that the model accurately predicts user ratings, allowing it to deliver recommendations that align well with user preferences.

## Future Improvements
- **Hybrid Recommendation Approach**: Combine collaborative filtering with content-based filtering to improve recommendation accuracy.
- **Model Improvement**: introduce more sophisticated robustness techniques or apply neural network recommendations
- **Interactive User Interface**: Create a simple web or desktop interface for users to input ratings and receive recommendations.
- **Real-Time Updates**: Integrate real-time data handling to dynamically update recommendations as users provide new ratings.
