class dnodestruct:
    # This class defines the tree structure
    def __init__(self, atr=-1, value=None, results=None, lcb=None, rcb=None, current_depth=0, needed_depth=None):
        self.atr=atr # column index of criteria being tested
        self.value=value # value necessary to get a true result
        self.results=results # dict of results for a branch, None for everything except endpoints
        self.lcb=lcb # true decision nodes - Left child branch
        self.rcb=rcb # false decision nodes - Right child branch
        self.current_depth = current_depth
        self.needed_depth = needed_depth


# Returns two datasets by dividing the rows by the given index and value
def divideDataset(rows,idx,value):
    divide_function = None
    # Numerical Values
    lset = []
    rset = []
    if (type(value) == int or type(value) == float):
        for row in rows:
            if(row[idx] >= value):
                lset.append(row)
            else:
                rset.append(row)
    else:
        for row in rows:
            if(row[idx] == value):
                lset.append(row)
            else:
                rset.append(row)


    return [lset,rset]


# Determine whether the given row is classified into the true branch or false branch of a given node.
def decideAttr(row,column,value):

    decide_function = lambda row:True if row[column] == value else False

    if decide_function(row):
        return True
    else:
        return False


# Get the list of classes along with their respective number of occurrences in the dataset.
def unique_count(rows):
    results={}
    for row in rows:
        # The result is the last column
        r=row[-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results


# Entropy is the sum of p(x)log(p(x)) across all the different possible results
def entropy(rows):
    
    
    results = unique_count(rows)
    ent = 0.0
    for r in results.keys():
        # current probability of class
        p = float(results[r])/len(rows)
        ent = ent-p*log_computation(p)
    return ent

def log_computation(x):
    from math import log
    log2=log(x)/log(2)
    return(log2)

def buildtree(rows,current_depth,needed_depth):

    if(current_depth == needed_depth):
        return dnodestruct(results=unique_count(rows))
    

    if len(rows) == 0: return dnodestruct()
    current_score = entropy(rows)

    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0])- 1	# last column is result
    for col in range(0, column_count):
        # find different values in this column
        column_values = set([row[col] for row in rows])
       
        # for each unique value, divide the set on that value
        for value in column_values:
            set1, set2 = divideDataset(rows, col, value)

            # Calculate the gain and determine the best sets.
            p = float(len(set1)) / len(rows)
            gain = current_score - p*entropy(set1) - (1-p)*entropy(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    
    return bestGain(best_gain,best_sets,current_depth,needed_depth,best_criteria,rows)   
    

def decision(tree,data):
    current = tree
    while current.results == None:
        if decideAttr(data,current.atr,current.value):
            current = current.lcb
        else:
            current = current.rcb
    return current.results

def bestGain(best_gain,best_sets,current_depth,needed_depth,best_criteria,rows):
    if best_gain > 0:
        leftBranch = buildtree(best_sets[0], current_depth+1, needed_depth)
        rightBranch = buildtree(best_sets[1], current_depth+1, needed_depth)


        var1=dnodestruct(atr=best_criteria[0], value=best_criteria[1],
                lcb=leftBranch, rcb=rightBranch, current_depth=current_depth, needed_depth=needed_depth)
        
    else:
        var1= dnodestruct(results=unique_count(rows))
    return var1
