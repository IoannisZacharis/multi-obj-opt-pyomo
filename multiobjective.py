#Solving multi-objective non-linear optimization  problem using pyomo model with gurobi solver
#


import numpy as np
import math
import  csv, random
from inspect import currentframe
import gurobipy
from pyomo.environ import *




#function that provides 2d samle, random:optionally 
def get2dSample(inptList : list, 
                    rowSize : int,
                    colSize : int,
                    random_ : bool = False) -> list:

        return [random.sample(row, colSize) for row in \
                random.sample(inptList, rowSize)] if random_ else \
                [row[:colSize] for row in inptList[:rowSize]]

#importing the csv files and convert them to 2d list,you have to import thre filepath
data = list(csv.reader(open("C:\\Users\\Dell\\Downloads\\NFMOLE.csv"), delimiter=";")
                   )
#data =get2dSample(data, 
                             #   rowSize = 500, 
                             #  colSize = 500)
#Importing thr weight csv, filepath as string                              
weights = list(csv.reader(open( "C:\\Users\\Dell\\Downloads\\weights.csv"), delimiter=";")
                      )

#Convertiing data to binary ,depending on threashold 
riskD = [
                [1 if float(val) >= .072 else 0 for val in row] \
                    for row in data
                ]
#This function imports the weights for each column of the data matrix and returns the new updated matrix and the total sum of weights (max_weigth_risk) 
def importing_weights(weights:list)->list[list] and float:
    weights = [float(k) for k in sum(weights, [])]
    weights = weights[:len(data[0])]
        # weights = [1 for _ in weights_]
    weights = [k/2 for k in weights]

    return[
        [riskD[r][v] * weights[v] for v in range(len(riskD[r]))]\
        for r in range(len(riskD))
        ], sum(weights)


#This function is responsible for improving the 2d data , by reordering the rows of the matrix depending on the sum of the data on each row
def improve(A: list[list]) -> list[list]:
    sums = {}

    for i in range(len(A)):
        if sum(A[i]) in sums:
            sums[sum(A[i])].append(i)
        else:
            sums[sum(A[i])] = [i]

    B = []
    for key in sorted(sums.keys(), reverse=True):
        for i in sums[key]:
            B.append(A[i])

    return B
   
    

#this function use is for creating the logic that minimizes the len of the set of selected_rows witch is the first variable of the model  
def algorithm(A: list[list], c: int) -> int or set or float:
    
    columns = [i for i in range(len(A[0]))]
    total_data_weighted_sum = 0
    selected_rows = set()

    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] !=0 and j in columns:
                total_data_weighted_sum+=A[i][j]
                selected_rows.add(i)
                columns.remove(j)
    
    rows_selected = len(selected_rows)
    current_percentage = total_data_weighted_sum / total_weight_list_sum

    if c == 1:
        return rows_selected
    elif c == 2:
        return selected_rows
    elif c == 3:
        return current_percentage
    elif c==4:
        print(total_data_weighted_sum)

#initializing the pyomo model 
def optimization(A: list[list], target_percentage: float)->int and float and set:
    #Create the model
    model = ConcreteModel()
    #Declare variables 
    model.rows_selected = Var(bounds=(0, len(new_set_of_data)), within=NonNegativeIntegers, name='selected_rows')
    model.current_percentage = Var(bounds=(0, len(new_set_of_data[0])), within=NonNegativeReals, name='current_percentage')
    #Declare the objective functions
    model.obj1 = Objective(expr=model.rows_selected, sense=minimize)
    model.obj2 = Objective(expr=model.current_percentage, sense=maximize)
    
    #Declare the model constraints 
    model.con1 = Constraint( expr=algorithm(new_set_of_data, 3) == model.current_percentage )
    model.con2 = Constraint(expr=model.current_percentage >= target_percentage)
    model.con3 = Constraint(expr=model.rows_selected == algorithm(new_set_of_data, 1))

    opt = SolverFactory ('gurobi') 
    model.obj2.deactivate() 
    results = opt.solve(model)
    #Model Display
    print('rows_selected = ', round(value(model.rows_selected), 2))
    print('current_percentage = ', round(value(model.current_percentage), 2))
    print('obj1 = ', round(value(model.obj1), 2))
    print('obj2 = ', round(value(model.obj2), 2))
    return value(model.obj1), value(model.obj2), algorithm(A, 2),results

if __name__ == "__main__":
    target_percentage = 0.9
    new_set_of_data, total_weight_list_sum=importing_weights(weights)
    new_set_of_data=improve(new_set_of_data) 
    a, b, d ,e = optimization(new_set_of_data, target_percentage=target_percentage)
    print(a,b,d,e)
    #print(0.9*total_weight_list_sum)
    #print(algorithm(new_set_of_data,4))
    
    
   