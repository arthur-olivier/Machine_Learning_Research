import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics





# Loads GloVe embeddings from a designated file location. 

#
# Invoked via:
# ge = GloVe_Embedder(path_to_embeddings)
#
# Embed single word via:
# embed = ge.embed_str(word)
#
# Embed a list of words via:
# embeds = ge.embed_list(word_list)
#
# Find nearest neighbors via:
# ge.find_k_nearest(word, k)
#
# Save vocabulary to file via:
# ge.save_to_file(path_to_file)

class GloVe_Embedder:
    def __init__(self, path):
        self.embedding_dict = {}
        self.embedding_array = []
        self.unk_emb = 0
        # Adapted from https://stackoverflow.com/questions/37793118/load-pretrained-GloVe-vectors-in-python
        with open(path,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                self.embedding_dict[word] = embedding
                self.embedding_array.append(embedding.tolist())
        self.embedding_array = np.array(self.embedding_array)
        self.embedding_dim = len(self.embedding_array[0])
        self.vocab_size = len(self.embedding_array)
        self.unk_emb = np.zeros(self.embedding_dim)

    # Check if the provided embedding is the unknown embedding.
    def is_unk_embed(self, embed):
        return np.sum((embed - self.unk_emb) ** 2) < 1e-7
    
    # Check if the provided string is in the vocabulary.
    def token_in_vocab(self, x):
        if x in self.embedding_dict and not self.is_unk_embed(self.embedding_dict[x]):
            return True
        return False

    # Returns the embedding for a single string and prints a warning if
    # the string is unknown to the vocabulary.
    # 
    # If indicate_unk is set to True, the return type will be a tuple of 
    # (numpy array, bool) with the bool indicating whether the returned 
    # embedding is the unknown embedding.
    #
    # If warn_unk is set to False, the method will no longer print warnings
    # when used on unknown strings.
    def embed_str(self, x, indicate_unk = False, warn_unk = True):
        if self.token_in_vocab(x):
            if indicate_unk:
                return (self.embedding_dict[x], False)
            else:
                return self.embedding_dict[x]
        else:
            if warn_unk:
                    print("Warning: provided word is not part of the vocabulary!")
            if indicate_unk:
                return (self.unk_emb, True)
            else:
                return self.unk_emb

    # Returns an array containing the embeddings of each vocabulary token in the provided list.
    #
    # If include_unk is set to False, the returned list will not include any unknown embeddings.
    def embed_list(self, x, include_unk = True):
        if include_unk:
            embeds = [self.embed_str(word, warn_unk = False).tolist() for word in x]
        else:
            embeds_with_unk = [self.embed_str(word, indicate_unk=True, warn_unk = False) for word in x]
            embeds = [e[0].tolist() for e in embeds_with_unk if not e[1]]
            if len(embeds) == 0:
                print("No known words in input:" + str(x))
                embeds = [self.unk_emb.tolist()]
        return np.array(embeds)
    
    # Finds the vocab words associated with the k nearest embeddings of the provided word. 
    # Can also accept an embedding vector in place of a string word.
    # Return type is a nested list where each entry is a word in the vocab followed by its 
    # distance from whatever word was provided as an argument.
    def find_k_nearest(self, word, k, warn_about_unks = True):
        if type(word) == str:
            word_embedding, is_unk = self.embed_str(word, indicate_unk = True)
        else:
            word_embedding = word
            is_unk = False
        if is_unk and warn_about_unks:
            print("Warning: provided word is not part of the vocabulary!")

        all_distances = np.sum((self.embedding_array - word_embedding) ** 2, axis = 1) ** 0.5
        distance_vocab_index = [[w, round(d, 5)] for w,d,i in zip(self.embedding_dict.keys(), all_distances, range(len(all_distances)))]
        distance_vocab_index = sorted(distance_vocab_index, key = lambda x: x[1], reverse = False)
        return distance_vocab_index[:k]

    def save_to_file(self, path):
        with open(path, 'w') as f:
            for k in self.embedding_dict.keys():
                embedding_str = " ".join([str(round(s, 5)) for s in self.embedding_dict[k].tolist()])
                string = k + " " + embedding_str
                f.write(string + "\n")

#---------------------------------------------------------GLOBAL PCA----------------------------------------
'''
path='GloVe_Embedder_data.txt'
ge = GloVe_Embedder(path)
list_word=['flight','good','terrible','help','late']
#list_word=['flight']
perplexity=28

for word in list_word:
        
    word_in_the_vocabulary=ge.token_in_vocab(word)
    embed = ge.embed_str(word)
    similar_word_general=ge.find_k_nearest(embed, 29)
    similar_word = [word[0] for word in similar_word_general]
    similarity = [word[1] for word in similar_word_general] 
    similar_embed=ge.embed_list(similar_word)
    # print("##############")
    # print('FOR WORD : ',word )
    # print("##############")
    # print(similar_word)

    pca1 = PCA(n_components=2)
    pca = pca1.fit_transform(similar_embed)
    X=pca[:,0]
    Y=pca[:,1]

    scatter=plt.scatter(X, Y)

    # IF we want to plot the text for each points 
    for i, label in enumerate(similar_word):
       plt.annotate(label, (X[i], Y[i]))

plt.legend(list_word)
plt.show()
'''
#---------------------------------------------------------GLOBAL TSNE----------------------------------------
'''
path='GloVe_Embedder_data.txt'
ge = GloVe_Embedder(path)
list_word=['flight','good','terrible','help','late']
#list_word=['flight']
perplexity_liste=[5,10,15,20,25,30,35,40,45,50]
for perplexity in perplexity_liste:

    for word in list_word:
            
        word_in_the_vocabulary=ge.token_in_vocab(word)
        embed = ge.embed_str(word)
        similar_word_general=ge.find_k_nearest(embed, 29)
        similar_word = [word[0] for word in similar_word_general]
        similarity = [word[1] for word in similar_word_general] 

        similar_embed=ge.embed_list(similar_word)
        tsne1 = TSNE(n_components=2,perplexity=perplexity,metric='euclidean')
        tsne = tsne1.fit_transform(similar_embed)
        X=tsne[:,0]
        Y=tsne[:,1]

        scatter=plt.scatter(X, Y)

    plt.title("t-SNE for Perplexity: " + str(perplexity))
    plt.legend(list_word)
    plt.axis([-1000, 1000, -1000, 1000])
    savefile='perplexity :' + str(perplexity)
    plt.savefig(savefile)

'''
#---------------------------------------------------------GLOBAL TSNE----------------------------------------

path='GloVe_Embedder_data.txt'
ge = GloVe_Embedder(path)
list_word=['flight','good','terrible','help','late']
#list_word=['flight']
k=2
all_words=[]
similarity_total=[]

for word in list_word:
        
    word_in_the_vocabulary=ge.token_in_vocab(word)
    embed = ge.embed_str(word)
    similar_word_general=ge.find_k_nearest(embed, 30)
    similar_word = [word[0] for word in similar_word_general]
    similarity = [word[1] for word in similar_word_general] 
    all_words+=similar_word
    similarity_total+=similarity

k_liste=[]
liste=[]
true_label=[]
for j in range(0,5):
    for i in range(0,30):
        true_label+=[j]

for i in range(2,20):
    k=i
    similar_embed=ge.embed_list(all_words)
    kmeans1 = KMeans(n_clusters=k)
    kmeans = kmeans1.fit(similar_embed)
    inertia=kmeans.inertia_
    
    k_liste+=[k]
    print('############ . ', k)


    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

    score=adjusted_rand_score(true_label,kmeans.labels_)
    liste+=[score]
    print(score)
    
    
   
# print(similarity_total)
# print(len(kmeans.labels_))
#normalized_mutual_info_score
#adjusted_rand_score

X=k_liste
Y=liste
plt.xlabel('k value')
plt.ylabel('adjusted rand score')
plt.scatter(X, Y)
savefile='adjusted_rand_score'
plt.savefig(savefile)

