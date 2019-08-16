# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:43:08 2019

@author: SSB
"""

import networkx as nx
import sys
import numpy as np

def constructLaplacianMatrix(list_cluster, G):
    clusterListLength = len(list_cluster)
    list_cluster = sorted(list_cluster)
    
    adjacencyMatrix = np.zeros((clusterListLength, clusterListLength))
    for node in list_cluster:
        for adjecentNode in list(G.adj[node]):
            if adjecentNode in list_cluster:
                adjacencyMatrix[list_cluster.index(node)][list_cluster.index(adjecentNode)] = 1
                adjacencyMatrix[list_cluster.index(adjecentNode)][list_cluster.index(node)] = 1
    degreeMatrix = np.zeros((clusterListLength, clusterListLength))
    for nodeIndex in range(clusterListLength):
        degreeMatrix[nodeIndex][nodeIndex] = G.degree[list_cluster[nodeIndex]]

    laplacianMatrix = degreeMatrix - adjacencyMatrix   
    return laplacianMatrix

def removeEdgesFromGraph(list_clusterForPositiveEigenVectorPart, list_clusterForNegativeEigenVectorPart, G):
    for node in list_clusterForPositiveEigenVectorPart:
        for adjacentNode in list(G.adj[node]):
            if adjacentNode in list_clusterForNegativeEigenVectorPart:
                G.remove_edge(node, adjacentNode)
    return G

       
inputFile = open(sys.argv[1], "r")
k = int(sys.argv[2])

list_clusters = []
set_Nodes = set()
G = nx.Graph()
for row in inputFile.readlines(): 
    row = row.split()
    set_Nodes.add(int(row[0]))
    set_Nodes.add(int(row[1]))
    G.add_node(int(row[0]))
    G.add_node(int(row[1]))
    G.add_edge(int(row[0]), int(row[1]))
list_clusters.append(sorted(list(set_Nodes)))


for i in range(k - 1):
    # pick largest cluster 
    maximumClusterLength = float('-inf')
    list_largestCluster = []
    for c in list_clusters:
        if len(c) > maximumClusterLength:
            list_largestCluster = c
            maximumClusterLength = len(c)
    # print(list_largestCluster)    
    
    laplacianMatrix = constructLaplacianMatrix(list_largestCluster, G)
    # print(laplacianMatrix)
    list_clusters.remove(list_largestCluster)

    #construction of EigenVector and EigenValues
    eigenValues, eigenVectors = np.linalg.eig(laplacianMatrix)

    # print(eigenValues)
    # print(eigenVectors)
        
    # change : pick the second smallest eigenvalue and corresponding eigenvector
    minimum_pc1= float('inf')
    minimum_pc2 = float('inf')
    for eigenValue in eigenValues:
        if eigenValue <= minimum_pc1:
            minimum_pc1, minimum_pc2 = eigenValue, minimum_pc1
        elif eigenValue < minimum_pc2:
            minimum_pc2 = eigenValue

    eigenValueIndex = list(eigenValues).index(minimum_pc2)
    # print(eigenValueIndex)
    # print(eigenVectors[:, eigenValueIndex])
    secondSmallest_eigenVector = eigenVectors[:, eigenValueIndex]
        
    # split the eigenvector at 0 into 2 list_clusters
    list_clusterForPositiveEigenVectorPart = []
    list_clusterForNegativeEigenVectorPart = []
    for i in range(len(secondSmallest_eigenVector)):
        if secondSmallest_eigenVector[i] >= 0:
            list_clusterForPositiveEigenVectorPart.append(list_largestCluster[i])
        else:
            list_clusterForNegativeEigenVectorPart.append(list_largestCluster[i])
    list_clusters.append(sorted(list_clusterForPositiveEigenVectorPart))
    list_clusters.append(sorted(list_clusterForNegativeEigenVectorPart))
    G = removeEdgesFromGraph(list_clusterForPositiveEigenVectorPart, list_clusterForNegativeEigenVectorPart, G)
        

list_finalCluster = []
for cluster in list_clusters:
    cluster = sorted(cluster)
    # print(len(cluster))
    list_finalCluster.append(",".join(str(node) for node in cluster))


# print(list_finalCluster)


w = open(sys.argv[3], "w")
for row in list_finalCluster:
    w.write(row + "\n")
