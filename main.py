import numpy as np
import pandas as pd              #this is for "scanning" the databases
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  #use a method that help us to split the data to train and test
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt  #from this import and below ifs for the visuazlization part
import graphviz                  #so we import ready libaries that take our implemention and
import seaborn as sns            #creat the graphic tree and cofusion matrix
import networkx as nx



class DecisionTree:
     #constructor that accept maxDepth,minSamplesSplit(the min number of samples that require to split the node)
     #and minSamplesLeaf that requires the minimun number of leafs that should be to a node ,if less its a leaf
    def __init__(self, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1): 
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.tree = None
    
    #self=where be stored the tree x=features of the data set,y=label to each sample
    def fit(self, X, y):
        self.tree = self.buildDesicionTree(X, y, depth=0)
    
    #function that recursively build the tree
    def buildDesicionTree(self, X, y, depth, feature_index=None):
      nSamples, nFeatures = X.shape   #number of samples and features
      n_classes = len(np.unique(y))   #number of labels
      bestGini = 1.0                 #a var to keep track after the best gini impurity
      bestCriteria = None            #keep the result with best split
      bestSets = None                #keep the kids of the best split

    # stopping conditions=check the conditions like the maxDepth minSamples and stop the building of the tree
      if (depth == self.maxDepth or n_classes == 1 or nSamples < self.minSamplesSplit):
          return self.most_common_label(y)

    # iterate over all features(in matrix x)
      for feature_idx in range(nFeatures):
        # if feature_index is provided so keep the iteration
         if (feature_index is not None and feature_index != feature_idx):
              continue

        # check the data type of x to know in which colum to put that can handle the type
         if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, feature_idx]
         else:
            X_col = X[:, feature_idx]

        # Iterate over all unique values in the selected feature
         unique_values = np.unique(X_col)
         for threshold in unique_values:
            # Split the data into left and right sets 
            left_indices = X_col < threshold  #feature values that less than the threshold so be in the left "kids"
            right_indices = ~left_indices     #feature values that is greater or equal to the threshold be the right "kids"
            y_left = y[left_indices]
            y_right = y[right_indices]

            # Skip if the number of samples in left or right smaller than the minSamplesLeaf
            if len(y_left) < self.minSamplesLeaf or len(y_right) < self.minSamplesLeaf:
                continue

            # calculate gini impurity for the current split in the current iteration
            gini_left = self.caculateGini(y_left)
            gini_right = self.caculateGini(y_right)
            gini = (len(y_left) / nSamples) * gini_left + (len(y_right) / nSamples) * gini_right

            # update the best split value if the current split value (gini) is better
            if gini < bestGini:
                bestGini = gini
                bestCriteria = (feature_idx, threshold)
                bestSets = (left_indices, right_indices)

    # Recursively build the tree for left and right subsets
      if bestGini > 0:
          left_subtree = self.buildDesicionTree(X[bestSets[0]], y[bestSets[0]], depth + 1)
          right_subtree = self.buildDesicionTree(X[bestSets[1]], y[bestSets[1]], depth + 1)
      else:
        # bestGini<0, we stil will create leaf nodes for both subsets
          left_subtree = self.most_common_label(y[bestSets[0]])
          right_subtree = self.most_common_label(y[bestSets[1]])

      return ( bestCriteria, left_subtree, right_subtree)
    
    #check if the desicion tree trained or not,if the tree is not none so it call for each of the samples in X to predcitOne
    def predict(self, X):
        if self.tree is not None:
            return np.array([self.predictEachSample(x, self.tree) for x in X])
        else:
            raise ValueError("The decision tree is not trained yet.")
    
    #predict for each sample,passes over the tree from the root and when we reach the node we return 
    def predictEachSample(self, x, node):
        if isinstance(node, tuple):
            feature_index, threshold = node[0]
            left_subtree, right_subtree = node[1], node[2]
            if x[feature_index] < threshold:
                return self.predictEachSample(x, left_subtree)
            else:
                return self.predictEachSample(x, right_subtree)
        else:
            return node
    
    #return the most commom label in y ,help for predictions
    def most_common_label(self, y):
        return np.bincount(y).argmax()
    
    #calculates the gini impurity for  y(1d array of labels)
    def caculateGini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)
    
class RandomForest:
    #constructor for the random forest class,contaian num of tress,maxDepth,minSamplesSplit
    #minSamplesLeaf and empty randomforest list that will contain all the desicion tress in the forest
    def __init__(self, nTress=30, maxDepth=None, minSamplesSplit=2, minSamplesLeaf=1):
        self.nTress = nTress   #number of tress,in default 30
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.random_forest = []
    
    #send each tree to the desicion tree to train and afte that add to the forest
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):  #chech the data type
            X = X.values  
        if isinstance(y, pd.Series):
            y = y.values  

        for _ in range(self.nTress):
            tree = DecisionTree(maxDepth=self.maxDepth, minSamplesSplit=self.minSamplesSplit, minSamplesLeaf=self.minSamplesLeaf)
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.random_forest.append(tree)
    
    #make predictions on the X matrix(features)
    def predict(self, X):
        if len(self.random_forest) == 0:
            raise ValueError("The random forest is not trained yet.")
        
        # make predictions using each tree in the forest
        predictions = [tree.predict(X) for tree in self.random_forest]
        
        # Use majority voting to make the final prediction(the most common predicition)
        y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)
        return y_pred
    
    #predint for a single sample each time by passing all over the tree
    def predictEachSample(self, x, node):
        if isinstance(node, tuple):
            feature_index, threshold = node[0]
            left_subtree, right_subtree = node[1], node[2]
            if x[feature_index] < threshold:
                return self.predictEachSample(x, left_subtree)
            else:
                return self.predictEachSample(x, right_subtree)
        else:
            return node

 #this class responsible to the visualization of the trees     
class TreeVisualizer:
    #constructor that accept names of the features and labels
    def __init__(self, feature_names, labels_names):
        self.feature_names = feature_names
        self.labels_names = labels_names

    def visualizeDecisionTrees(self, trees):
        for i, tree in enumerate(trees):
            if "DecisionTree" in str(type(tree)):
                self.visualizeTree(tree.tree, depth=0)  # Start with depth 0
            else:
                # we use scikit learn decision tree to visualize
                dot_data = tree.export_graphviz(
                    feature_names=self.feature_names,
                    labels_names=self.labels_names,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                )
                graph = graphviz.Source(dot_data)
                graph.render(f"decision_tree_{i + 1}")

    def visualizeTree(self, node, depth, parent=None, edge_type=None):
        # Create a graph using networkx for visualization
        if parent is None:
            graph = nx.DiGraph()
            root_node = "Root"
            graph.add_node(root_node)
        else:
            graph = parent[0]
            root_node = parent[1]

        if isinstance(node, tuple):
            feature_idx, threshold = node[0]
            left_subtree, right_subtree = node[1], node[2]

            # add nodes for the current decision node and edges from the parent
            current_node = f"Depth {depth}: {self.feature_names[feature_idx]} < {threshold}"
            graph.add_node(current_node)
            if edge_type is not None:
                graph.add_edge(edge_type, current_node)

            # recursively visualize the left and right subtrees
            self.visualizeTree(left_subtree, depth + 1, parent=(graph, current_node), edge_type="Left")
            self.visualizeTree(right_subtree, depth + 1, parent=(graph, current_node), edge_type="Right")
        else:
            # leaf node, print the predicted class
            predicted_class = int(node)  # Convert the predicted class to an integer
            class_name = self.labels_names[predicted_class]
            leaf_node = f"Leaf: class {class_name}"
            graph.add_node(leaf_node)
            if edge_type is not None:
                graph.add_edge(edge_type, leaf_node)

        # if this is the root node, show the graph(cause it mean we have all the tree)
        if parent is None:
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=1500, node_color='green')
            plt.show()     

def main():
    file_path = "iris.data"
    #file_path="heart.csv"
    df = pd.read_csv(file_path, header=None)
    feature_names = list(df.columns[:-1])  # the last column is the label column

    # Separate features (X) and labels (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    label_encoder = LabelEncoder() #encode lables to numrical so we can work with this
    y = label_encoder.fit_transform(y)
    labels_names = label_encoder.classes_  #return to the original

    # divide into training and test sets with the train_test_spilt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=48)

    # convert the training data to arrays if needed
    X_trainArray = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_trainArray = y_train.values if isinstance(y_train, pd.Series) else y_train

    # create and train the decision tree(sent to the desicionTree class)
    #we can use different maxDepth,minSamplesLeaf and minSamplesSplit ofcourse
    decision_tree = DecisionTree(maxDepth=3, minSamplesSplit=5, minSamplesLeaf=2)
    decision_tree.fit(X_trainArray, y_trainArray)

    # Create and train the random forest(sent to the randomForest class)
    random_forest = RandomForest(nTress=10, maxDepth=3, minSamplesSplit=5, minSamplesLeaf=2)
    random_forest.fit(X_trainArray, y_trainArray)

    # predict using the random forest
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_pred_ensemble = random_forest.predict(X_test_array)

    # predictions by each tree in the random forest
    individual_tree_predictions = []    #to here we will add
    for i, tree in enumerate(random_forest.random_forest):
        y_pred_tree = tree.predict(X_test_array)
        individual_tree_predictions.append(y_pred_tree)
        print(f"Predictions by Tree {i + 1}:", y_pred_tree)

    # Final enpredictions
    print("Ensemble Predictions:", y_pred_ensemble)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    print("The accuracy is: ",accuracy)

    # calculate confusion matrix
    confusion = confusion_matrix(y_test, y_pred_ensemble)

     #Visualize each tree in the random forest
    tree_visualizer = TreeVisualizer(feature_names, labels_names)
    for i, tree in enumerate(random_forest.random_forest):
        print(f"Visualizing Decision Tree {i + 1}")
        tree_visualizer.visualizeDecisionTrees([tree])

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, cmap="YlGn", fmt="d", xticklabels=labels_names, yticklabels=labels_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show() 

if __name__ == "__main__":
    main()
