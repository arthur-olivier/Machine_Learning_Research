# Word Embeddings and Sentiment Analysis

This project aims to explore unsupervised learning techniques using pre-trained word embeddings and improve sentiment classification performance through the use of word embeddings.

# Data Set

Please note that the data set used for training and testing the model is not included in this repository. The two data sets required to run the code are 'IA3-train.csv" and "IA3-dev.csv", and each is composed of two columns: one with the sentiment (0 or 1) and the second column with the text of the tweet.

You will need to provide your own data set in the same format if you want to use the code in this repository. Make sure that the file names and column names match exactly with the ones specified above. The first column has to be "sentiment" and the second column "text".

If you have any questions or issues regarding the data set, please feel free to contact me.

# Data

The GloVe word embedding is used in this project, which is a popular word embedding that is pre-trained using large amounts of unlabeled text. To avoid dealing with the full size of the GloVe embeddings, a reduced subset of GloVe embeddings is provided, which corresponds to the intersection of the Twitter sentiment dataset’s vocabulary and the full GloVe vocabulary. A file named "GloVe Embedder data.txt" contains the embeddings, and "GloVe Embedder.py" is used to handle loading the embeddings and finding nearest neighbors.

# Explore Word Embeddings

##Build Data Set
A data set of 150 words is built to visualize and play with. The initial set of seed words is "flight", "good", "terrible", "help", and "late". For each seed word, the 29 most similar words are found based on the word embedding using Euclidean distance, resulting in five clusters. The 29 most similar words for each seed word are listed in the report.

##Dimension Reduction and Visualization
Different dimension reduction and visualization techniques are explored in this part.

1. PCA is applied to the 150 words, and the results are visualized in a 2-D space. Each seed word (and the words similar to that seed word) is assigned a distinct color. The resulting visualization shows five distinct clusters.
2. t-SNE is applied to the 150 words, and the results are visualized in a 2-D space using the same color mapping. Different perplexity values are explored, and the resulting impact on the visualization is observed. The report provides substantially different visualization results by using different perplexity parameters.

## Clustering
K-means clustering is applied to the words using different k values ranging from 2 to 20. For each k value, the resulting k-means objective is recorded, which measures the sum of squared distances between each point and its cluster center. The k-means objective is plotted as a function of k, and the resulting curve shows that the objective value monotonically decreases as we increase k. However, there is no evidence from this curve to suggest that k=5.

Using the original seed words as ground truth labels for clustering, the clustering solution is evaluated for different k values using different metrics, including purity, adjusted rand index, and normalized mutual information. The resulting metric scores are plotted as a function of k, and the report shows that k=5 does not necessarily give the best score for different metrics.

# Conclusion

In conclusion, this project explores unsupervised learning techniques using pre-trained word embeddings and shows how they can be used to improve sentiment classification performance. The report provides clear and concise figures and tables with necessary legends and captions. The resulting project is professional and can be presented as a standalone project rather than just an assignment.
