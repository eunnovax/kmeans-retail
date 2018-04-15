import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

values =[]
cid = []

def euclidian(a, b):
    return np.linalg.norm(a-b)

# Step1 - parsing value and customer id
with open('retail.csv','r') as csvfile:
	csvfilereader = csv.reader(csvfile, delimiter=';')
	next(csvfilereader)
	i = -1
	for row in csvfilereader:
		i += 1
		
		values.append(float(row[5]))
		cid.append(int(float(row[6])))
		
		#if not cid[i]:
		#	print('i=', i)

#Step2 - counting frequency of purchases
df = pd.DataFrame({'aa':cid})

index, counts = np.unique(df.values,return_counts=True)


sum_value=[]

# Step 3 - adding total price value of each index
j = -1
for indice in index:
	j += 1
	indice_value = [values[i] for i in range(len(cid)) if 
	cid[i] == indice]
	#print(indice_value)
	sum_value.append(sum(indice_value))

#Step 4 - log transform & converting vectors to a matrix of dataset
#print(index)
#print(sum_value)
#print(counts)
logsum = np.log(sum_value)
logcounts = np.log(counts)

dataset = np.column_stack((logcounts, logsum))

# Step 5 - plotting total value vs frequency graph
#plt.plot(logcounts,logsum, 'r'+'o') 
#plt.show()

# Step 6 - kmeans function definition
def kmeans(k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    num_instances, num_features = dataset.shape
    #intitiating centroids as random indexes of dataset
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    #history of centroids
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    # assign dataset points to a centroid
    belongs_to = np.zeros((num_instances, 1))
    # euclidean distance b/w old and new centroids
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    #stop iteration when centroid location doesn't change
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        #extract index from the dataset and iterate over it
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            #extract index of each centroid and iterate over it
            for index_prototype, prototype in enumerate(prototypes):
            	#norm distance b/w a point in dataset and a centroid
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)
            # find a centroid closest to the given point
            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))
        cluster = []
        for index in range(len(prototypes)):
        	# dataset point closest to a particular centroid
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            cluster.append(instances_close)
            #print(instances_close)
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            tmp_prototypes[index, :] = prototype


        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)

    # plot(dataset, history_centroids, belongs_to)

    return prototypes, cluster

def plot(prototypes, logsum, logcounts):
	colors = ['r', 'g']
	plt.plot(logcounts,logsum, 'r'+'o')
	plt.plot(prototypes[:,0],prototypes[:,1],'b'+'o')
	plt.show() 


def execute():
    
    centroids, cluster = kmeans(4)
    #plot(centroids, logsum, logcounts)
    dataplot1 = dataset[cluster[0]]
    dataplot2 = dataset[cluster[1]]
    plt.plot(dataplot1[:,0],dataplot1[:,1], 'g'+'o')
    plt.plot(dataplot2[:,0],dataplot2[:,1], 'b'+'o')
    dataplot3 = dataset[cluster[2]]
    plt.plot(dataplot3[:,0],dataplot3[:,1], 'r'+'o')
    dataplot4 = dataset[cluster[3]]
    plt.plot(dataplot4[:,0],dataplot4[:,1], 'r'+'o')

    plt.show()
execute()


