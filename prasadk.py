#! /usr/bin/python

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need. You
will also need to make sure to bring your decision tree learner in somehow
either by copying your code into a learn_decision_tree function or by importing
your decision tree code in from the file your wrote for PA#1. You will also need
some functions to handle creating your data bags, computing errors on your tree,
and handling the reweighting of data points.

For building your bags numpy's random module will be helpful.
'''

# This is the only non-native library to python you need
import decision_tree as dtree
import numpy 
import random
import csv
import os.path
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

'''
Function: load_and_split_data(datapath)
datapath: (String) the location of the UCI mushroom data set directory in memory

This function loads the data set. datapath points to a directory holding
agaricuslepiotatest1.csv and agaricuslepiotatrain1.csv. The data from each file
is loaded and returned. All attribute values are nomimal. 30% of the data points
are missing a value for attribute 11 and instead have a value of "?". For the
purpose of these models, all attributes and data points are retained. The "?"
value is treated as its own attribute value.

Two nested lists are returned. The first list represents the training set and
the second list represents the test set.
'''
def load_data(datapath,depth,bags,test_data,train_data):
	 if not os.path.exists('datasets'):
		 os.makedirs('datasets')

	 if not os.path.exists('dataset_depth3_bag5'):
		 os.makedirs('dataset_depth3_bag5')

	 if not os.path.exists('dataset_depth3_bag10'):
		 os.makedirs('dataset_depth3_bag10')

	 if not os.path.exists('dataset_depth5_bag5'):
		 os.makedirs('dataset_depth5_bag5')

	 if not os.path.exists('dataset_depth5_bag10'):
		 os.makedirs('dataset_depth5_bag10')

	 selectfile_train = open(input("enter the train file name: "), 'rU')
	 reader_train = list(csv.reader(selectfile_train))
	 newlist = []
	 #cleaning the train data file by deleting one column and writing it to a new csv file
	 def train_file(item,idx):
		 temp = item[20]
		 item [-1] = temp
		 newlist.append(item)
		 b = numpy.array(newlist)
		 c = numpy.delete(b, numpy.s_[20,21],1)
		 c = numpy.delete(b,(0), axis=0)
		 numpy.savetxt("datasets/train1.csv",c,delimiter=',',fmt='%s')
		 print("writing the train file ",idx,"of 6000")
	 [train_file(item,idx) for idx,item in enumerate(reader_train)]
	 print("Done Writing the train file")

	 selectfile_test = open(input("enter the test file name: "), 'rU')
	 reader_test = list(csv.reader(selectfile_test))
	 newlist = []

	 #cleaning the test data file by deleting one column and writing it to a new csv file
	 def test_file(item,idx):
		 temp = item[20]
		 item [-1] = temp
		 newlist.append(item)
		 b = numpy.array(newlist)
		 c = numpy.delete(b, numpy.s_[20,21],1)
		 c = numpy.delete(b,(0), axis=0)
		 numpy.savetxt("datasets/test1.csv",c,delimiter=',',fmt='%s')
		 print("writing the test file ",idx,"of 6000")
	 [test_file(item,idx) for idx,item in enumerate(reader_test)]
	 print("Done Writing the test file")
	
	


'''
Function: learn_bagged(tdepth, numbags, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numbags: (Integer)the number of bags to use to learn the trees
datapath: (String) the location in memory where the data set is stored

This function will manage coordinating the learning of the bagged ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_bagged(depth, numbags, datapath,load_data,test_data):

	Accuracy_list=[]
	Depth_list = []



	depth = int(input("Please enter the depth:"))
	bags =  int(input("Please enter number of bags:"))


	#This method parses the file and returns the dataset
	def parse_file(file_path): 
		f = open(file_path, 'r')
		#print(f)
		file_data = []
		for line in f.readlines():
			line_strip = line.strip('\n').split(',')
			rowData = []
			for value in line_strip:
				try:
					rowData.append(float(value))
				except:
					rowData.append(value)
			file_data.append(rowData)
		f.close()
		return file_data


	def train_data(filename):
		# we are trying to input the Train, test file and the delimiter 
		train_file_path = filename.strip()
		if os.path.exists(train_file_path):
			return parse_file(train_file_path)


	def test_data_input():
		file_name = input('Enter the test file name: ')
		test_file_path = 'datasets/' + file_name.strip()
		if os.path.exists(test_file_path):
			return parse_file(test_file_path)    
	def buildtree(depth, test_data, train_data, current_index):


		tot_count = 0
		tot_correct = 0

		#Calculating the Accuracy at every level 
		correct = 0
		total =0
		TP = 0
		TN = 0
		FP = 0
		FN = 0
		#print ("Depth Entered is :" ,depth)
	 
		predicted_list = []
		predicted_list_1 = []
		for i in range (depth):

			tree = dtree.buildtree(train_data,0,i)
			for data in train_data:
				predicted = list(dtree.decision(tree,data).keys())[-1]
				predicted_list.append(predicted)

			for data in test_data:
				predicted = list(dtree.decision(tree,data).keys())[0]
				predicted_list_1.append(predicted)
				one_count_testdata = predicted_list_1.count(1)
				zero_count_testdata = predicted_list_1.count(0)
				actual = data[-1]
				total = total +1 
				if predicted == 1.0 and actual == 1.0:
					correct = correct + 1
					TP = TP + 1
				if predicted == 0.0 and actual == 0.0:
					correct = correct + 1
					TN = TN + 1
				if predicted == 1.0 and actual == 0.0:
					FP = FP+ 1
				if predicted == 0.0 and actual == 1.0:
					FN = FN + 1

			tot_correct += correct
			tot_count += total
			Accuracy = round(100*correct/total,2)
			Depth_list.append(depth)
			Accuracy_list.append(Accuracy)
			depth=depth+1
			#print (Accuracy_list)
			#print (Depth_list)
			#printing the confusion matrix
		print("Accuracy::",str(Accuracy)+'%')
		print("False Negatives ", str(FN))
		print("False positives ", str(FP))
		print("True Negatives ", str(TN))
		print("True Positives ", str(TP))
		print("Confusion Matrix for bagging")
		print("------")
		print("| ", TP , "|", FN, "|")
		print("------")
		print("| ", FP , "|", TN, "|")
		print("------")
		#plot_graph(current_index)  
			
	if depth == 3 and bags == 5:

		x = open('datasets/train1.csv','rU')
		reader = list(csv.reader(x))
		test_data = test_data_input()
		print('OPS ERROR')

		for i in range(0,5):
			b=numpy.random.permutation(reader)
			a = numpy.array(b)
			filename = "DATASET_for_depth_3_and_bags_5"+str(i)+".csv"
			numpy.savetxt("dataset_depth3_bag5/"+filename, a, delimiter=",", fmt="%s")
			traindata = train_data("dataset_depth3_bag5/"+filename)
			buildtree(depth, test_data, traindata, str(i+1))
			i = i+1
	else:
		if depth == 3 and bags ==10:
		 x = open('datasets/train1.csv','rU')
		 reader = list(csv.reader(x))
		 test_data = test_data_input()
		 for i in range(0,10):
			 b=numpy.random.permutation(reader)
			 a = numpy.array(b)
			 filename = "DATASET_for_depth_3_and_bags_10"+str(i)+".csv"
			 numpy.savetxt("dataset_depth3_bag10/"+filename, a, delimiter=",", fmt="%s")
			 traindata = train_data("dataset_depth3_bag10/"+filename)
			 buildtree(depth, test_data, traindata, str(i+1))
			 i = i+1

	if depth == 5 and bags == 5:
		x = open('datasets/train1.csv','rU')
		reader = list(csv.reader(x))
		test_data = test_data_input()
		for i in range(0,5):
			b=numpy.random.permutation(reader)
			a = numpy.array(b)
			filename = "DATASET_for_depth_5_and_bags_5_"+str(i)+".csv"
			numpy.savetxt("dataset_depth5_bag5/"+filename, a, delimiter=",", fmt="%s")
			traindata = train_data("dataset_depth5_bag5/"+filename)
			buildtree(depth, test_data, traindata, str(i+1))
			i = i+1
	else:
		if depth == 5 and bags ==10:
		 x = open('datasets/train1.csv','rU')
		 reader = list(csv.reader(x))
		 test_data = test_data_input()
		 for i in range(0,10):
			 b=numpy.random.permutation(reader)
			 a = numpy.array(b)
			 filename = "DATASET_for_depth_5_and_bags_10"+str(i)+".csv"
			 numpy.savetxt("dataset_depth5_bag10/"+filename, a, delimiter=",", fmt="%s")
			 traindata = train_data("dataset_depth5_bag10/"+filename)
			 buildtree(depth, test_data, traindata, str(i+1))
			 i = i+1


'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''
def learn_boosted(tdepth, numtrees, datapath):
	pass;


if __name__ == "__main__":
	# The arguments to your file will be of the following form:
	# <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
	# Ex. bag 3 10 mushrooms
	# Ex. boost 1 10 mushrooms

	# Get the ensemble type
	entype = sys.argv[1];
	train_data = sys.argv[1];
	test_data = sys.argv[1];
	# Get the depth of the trees
	depth = int(sys.argv[2]);
	# Get the number of bags or trees
	bags = int(sys.argv[3]);
	# Get the location of the data set
	datapath = sys.argv[4];

	# Check which type of ensemble is to be learned
	if entype == "bag":
		load_data(datapath,depth,bags,train_data,test_data);
		# Learned the bagged decision tree ensemble
		learn_bagged(depth, bags, datapath,load_data,test_data);
	else:
		# Learned the boosted decision tree ensemble
		learn_boosted(depth, bags, datapath);
