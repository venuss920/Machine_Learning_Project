##  SOURCE OF THE DATASET_1 - https://www.openml.org/search?type=data&status=active&id=42737&sort=runs
### Libraies used - https://scikit-learn.org/stable/, https://scipy.org/, https://matplotlib.org/, https://xgboost.readthedocs.io/en/stable/install.html
##  for xgboost algorithm-https://seaborn.pydata.org/

from sklearn.datasets import load_diabetes,fetch_california_housing
from xgboost import XGBRegressor,XGBClassifier
from cProfile import label
import pandas as pd
import numpy as np
from scipy.io import arff
from seaborn.regression import statsmodels 
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler,KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns



''' Checking and removing the missing values from the dataset_1 
grabbed from open ml or the features with object data types 
and considering only floatvalues for model training and testing  '''


##Loading the data set and conversion of the dataset for Frames per second -  dataset-1

path = 'C:\\Users\\Shah\\Desktop\\Studienarbeit\\fps_benchmark.arff'    #fps_data.arff
dataset_path,metadata = arff.loadarff(path)

#Conversion from arff to csv using pandas
dataframe = pd.DataFrame(dataset_path)
print(dataframe.head())
dataframe.to_csv('C:\\Users\\Shah\\Desktop\\Studienarbeit\\data_to_csv_copy_bench.csv',index=False)

#Reading the csv converted file and information extraction
path_1 = 'C:\\Users\\Shah\\Desktop\\Studienarbeit\\data_to_csv_copy_bench.csv'
data_converted = pd.read_csv(path_1)
print(data_converted.head())
print(data_converted.shape)                                             ###(425833,45)
print(data_converted.info())
ob_typs = data_converted.select_dtypes(include = 'object')

                                                                        #### 31 float types and 14 object types(string,names and etc)


#### find the missing values and try the adjustments
null_vals = data_converted.isnull().sum()
print(null_vals)

null_vals_in_cols = null_vals[null_vals > 0]
print(null_vals_in_cols,'Total_values_missing ', len(null_vals_in_cols))
print(null_vals_in_cols.index.tolist())

### Drop the columns with msing vals and save a copy of the dataframe
path_2 = 'C:\\Users\\Shah\\Desktop\\Studienarbeit\\data_to_csv_copy_bench.csv'
data_converted_copy = pd.read_csv(path_2)
store_data = data_converted_copy.dropna(axis = 1)
print(store_data)
store_data.to_csv('C:\\Users\\Shah\\Desktop\\Studienarbeit\\stored_without_null_vals_bench_data.csv',index = False)
print(store_data.shape)                                                ###(425833,45)
print(store_data.info())


#### Data stored without missing value columns
path_3 = 'C:\\Users\\Shah\\Desktop\\Studienarbeit\\stored_without_null_vals_bench_data.csv'
datas = pd.read_csv(path_3)
print(datas.shape)
print(datas.info())
print(datas.describe())

columns_ob_typ = datas.select_dtypes(include ='object').columns
datas_with_no_ob_typ = datas.drop(columns_ob_typ,axis =1)
datas_with_no_ob_typ.to_csv('C:\\Users\\Shah\\Desktop\\Studienarbeit\\stored_without_null_vals_and_no_ob_typ_bench.csv',index =False)





''' You can call these below functions in the for loop for regression algorithms 
 and classification algorithms to plot, save and 
  visualize the data charactersitics '''

### Passing the parameters in the function definitions and plotting the datas and visualizing for check the datasets.
### Function definitions fr data plot and visualisation.

def ploting(x,name):
    plt.figure(figsize=(10,10))
    sns.kdeplot(x,legend=True)
    #plt.show()
    plt.savefig(f'C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\X_{name}.png')


def plotting_bar(x):
    plt.figure(figsize=(6,6))
    sns.distplot(x,bins = 50,kde = True)
    plt.xlabel('Frames Per Second')
    plt.savefig('C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\Target_variable_and_plot_bar.png')
    #plt.show()



def plotting_bar_transform(x,i,t):
    plt.figure(figsize=(6,6))
    sns.distplot(x,bins = 50,kde = True)
    plt.savefig(f'C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\REGRESSION_RESULTS\\Target_variable_transform_and_plot_bar_{i}_{t}.png')
    #plt.show()
   

###Scateer plot for data visualisation original v/s predicted
def scatter_plots(data_test_y,data_predict_y,algorithm):
    print(data_test_y)
    print(data_predict_y)
    plt.figure(figsize=(6,6))
    
    plt.scatter(range(len(data_test_y)),data_test_y,color = 'blue',label='Test data')
    plt.scatter(range(len(data_predict_y)),data_predict_y,color = 'red',label='Predicted data')
    plt.title(f'Test data v/s predicted data {algorithm}')
    plt.xlabel('Splitted Test Data',ha = 'center')
    plt.ylabel('Frames Per Second',va = 'center')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\REGRESSION_RESULTS\\{algorithm}_figure.png')
    #plt.show()

### Check the residuals in the dataset original and prediction
def residual_plots(data_test_y,data_predict_y,algorithm):
    plt.figure(figsize=(5,5))
    plot_residuals = data_test_y-data_predict_y
    sns.kdeplot(plot_residuals)
    plt.title(f'Test data v/s predicted data {algorithm}')
    plt.grid(True)
    plt.savefig(f'C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\REGRESSION_RESULTS\\Residuals_{algorithm}_figure.png')
    #plt.show()




##### Plotting for the original doamin of regression dataset and the inverse transformation to check the predicitions
def scatter_plots_inverse(data_test_y,data_predict_y_inverse,algorithm,i,t):
    print(data_test_y)
    print(data_predict_y_inverse)
    plt.figure(figsize=(6,6))
    
    plt.scatter(range(len(data_test_y)),data_test_y,color = 'blue',label='Test data')
    plt.scatter(range(len(data_predict_y_inverse)),data_predict_y_inverse,color = 'red',label='Predicted data Inversion')
    plt.title(f'Test data v/s predicted inverse data {algorithm}')
    plt.xlabel('Splitted Test Data',ha ='center')
    plt.ylabel('Frames Per Second',va = 'center')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\RESULT_TRANSFORM\\INVERSE_{algorithm}_for_{i}_and_{t}_figure.png')
    #plt.show()

np.random.seed(42)


#### DATASETS Function definations

'''Note: For getting the results for every dataset, 
         Please uncomment on the dataset's function definition to be trained and test on, 
         with the its object initialisation 
         and comment the remaining ones'''


###### Training first the different regression model on different datasets
### DATASET-1 path from the directory.
path_4 = 'C:\\Users\\Shah\\Desktop\\Studienarbeit\\stored_without_null_vals_and_no_ob_typ.csv'



###  Function defination to extract datas from 

###  DATASET-1
def dataset_1(path):
    main_data = pd.read_csv(path)
    print(main_data)
   

    X = main_data.drop(columns=['FPS'])  ##Target Variable
    print(X)
    
    X = X.iloc[:, :].values
    y = main_data.iloc[:, -1].values
    
    print('datas description ',main_data.describe())
    return X,y

X,y = dataset_1(path_4)


### DATASET-2
def dataset_2(parameters):
    X = parameters.data
    y = parameters.target
    return X,y
Housing = fetch_california_housing()
X,y = dataset_2(Housing)    



### DATASET-3
def dataset_3(parameters_dataset_3):
    X = parameters_dataset_3.data
    y = parameters_dataset_3.target
    return X,y
dia = load_diabetes() 
X,y = dataset_2(dia)  


### Data splitting into training and testing
def splitting_datas_regression(X,y):            ### X,y depends on the dataset function definition uncommented for training either dataset_1, dataset,2 or dataset_3
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.75, test_size=0.25, random_state=42) 
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = splitting_datas_regression(X,y)


##Data pre-proccseing
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
print(X_test_scaled)





# Initialisation of the dictionary to store the key pairs and value pairs for regression results from training and testing
results_storing = {}
results_storing_training = {}






#### Tuning the hyperparameters
def cross_fold_params(n_splits, shuffle , random_state):
    cross__folds = KFold(n_splits = n_splits , shuffle = shuffle, random_state = random_state)
    return cross__folds


#### Hyperparameter tuning variables definition for Random Forest Regressor
random_forest_parameters = {
    'n_estimators': [10,50,70,100],
    'max_depth': [None, 5, 10]
}


search_parameters = GridSearchCV(estimator=RandomForestRegressor()
                           , param_grid = random_forest_parameters
                           , cv = cross_fold_params(5,True,42)
                           , scoring='neg_mean_squared_error')

search_parameters.fit(X_train_scaled, y_train)
print("Hyper parameters check best values for RANDOM_FOREST:",search_parameters.best_params_)
print("best score checking for RANDOM_FOREST:", -search_parameters.best_score_)  

####Model Training with accurate parameters 
rf_final = search_parameters.best_estimator_
print('RANDOM_FOREST : ',rf_final)

predictions_y = rf_final.predict(X_test_scaled)
print('RANDOM FORSET PREDICTIONS',predictions_y)

rmses = mean_squared_error(y_test,predictions_y,squared = False)
print('RMSES for random forest',rmses)





#### Hyperparameter tuning variables definition for Linear Regressor
linear_regression_parameters = {
    'fit_intercept': [True,False]
}

linear_regression_search_parameters = GridSearchCV(estimator= LinearRegression(), 
                                             param_grid=linear_regression_parameters, 
                                             cv=cross_fold_params(5,True,42), 
                                             scoring='neg_mean_squared_error')

linear_regression_search_parameters.fit(X_train_scaled, y_train)


print("Hyper parameters check best values for linear_regression:", linear_regression_search_parameters.best_params_)
print("best score checking for linear_regression:", -linear_regression_search_parameters.best_score_)


####Model Training with accurate parameters 
lr_final = linear_regression_search_parameters.best_estimator_
print('linear_regression',lr_final)

predictions_y_linear = lr_final.predict(X_test_scaled)
print('LINEAR REGRESSION PREDICTIONS',predictions_y_linear )

rmses_linear = mean_squared_error(y_test,predictions_y_linear,squared =False)
print('RMSES For Linear regression',rmses_linear)







#### Hyperparameter tuning variables definition for Decision Tree
DT_search_parameters = {
    'max_depth': [3,5,7,9,None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':[1,2,4,6],
    'max_features':['auto','sqrt','log2']
}
grid_search_dec_tree = GridSearchCV(estimator=DecisionTreeRegressor()
                                    , param_grid=DT_search_parameters
                                    , cv=cross_fold_params(5,True,42)
                                    , scoring='neg_mean_squared_error')

grid_search_dec_tree.fit(X_train_scaled, y_train)
print("Best Hyperparameters for DT:", grid_search_dec_tree.best_params_)
print("Best Score for DT :", -grid_search_dec_tree.best_score_) 

####Model Training with accurate parameters 
dec_tree_final_model = grid_search_dec_tree.best_estimator_
print('DT : ',dec_tree_final_model )

predictions_y_decision= dec_tree_final_model.predict(X_test_scaled)
print('DECISION TREE regression',predictions_y_decision)

rmses_decision = mean_squared_error(y_test,predictions_y_decision,squared =False)
print('MSES For DECISION TREE regression',rmses_decision)







#### Hyperparameter tuning variables definition for Gradient Boosting
GB_search_parameters = {
    'n_estimators': [10,50,100],
    'learning_rate': [0.001,0.01,0.1,0.5],
    'max_depth':[3,5,7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':[1,2,4,6]
    
}

grid_search_gd_boost = GridSearchCV(estimator=GradientBoostingRegressor(), 
                                    param_grid=GB_search_parameters, 
                                    cv=cross_fold_params(5,True,42), 
                                    scoring='neg_mean_squared_error')
grid_search_gd_boost.fit(X_train_scaled, y_train)

print("Best Hyperparameters for Gradient Boost:", grid_search_gd_boost.best_params_)
print("Best scores for Gradient boost algo:", -grid_search_gd_boost.best_score_) 

####Model Training with accurate parameters 
gd_boost = grid_search_gd_boost.best_estimator_
print('Gradient_Boost : ',gd_boost)

predictions_y_gradient= gd_boost.predict(X_test_scaled)
print('Gradient boosting PREDICTIONS',predictions_y_gradient)

rmses_gradient = mean_squared_error(y_test,predictions_y_gradient,squared =False)
print('RMSES For Gradient regression',rmses_gradient)






#### Hyperparameter tuning variables definition for KNeighbours
KNeighbours_parameters = {
    'n_neighbors': [5],
    'weights': ['uniform','distance'],
    'algorithm':['auto','ball_tree','kd_tree','brute'],
    'leaf_size':[30],
     'p': [1,2],
    }


KNN_search = GridSearchCV(estimator=KNeighborsRegressor()
                          , param_grid=KNeighbours_parameters
                          , cv=cross_fold_params(5,True,42)
                          ,scoring='neg_mean_squared_error')
KNN_search.fit(X_train_scaled, y_train)
print("Accurate Hyperparameters search for KNN : ", KNN_search.best_params_)
print("Accurate Score checking for KNN:", -KNN_search.best_score_) 

####Model Training with accurate parameters 
KNN_model = KNN_search.best_estimator_
print('KNN: ' , KNN_model)

predictions_y_KNN= KNN_model.predict(X_test_scaled)
print('KNN PREDICTIONS',predictions_y_KNN)

rmses_KNN = mean_squared_error(y_test,predictions_y_KNN,squared =False)
print('RMSES For Kneighbours',rmses_KNN)






### For dataset-1 Hyperparameters
Algorithms_regression_1 ={'Linear' : LinearRegression(fit_intercept=True), 
                        
                        'random_forest' : RandomForestRegressor(n_estimators= 100,max_depth = 10,random_state=42),
                        
                        'decision_tree' : DecisionTreeRegressor(max_features='log2', min_samples_leaf=6,
                         min_samples_split=10,random_state=42),
                        
                        'xgboost':XGBRegressor(objective ='reg:squarederror',random_state = 42) ,
                        
                        'gradient_boost': GradientBoostingRegressor(n_estimators = 100,learning_rate = 0.1,max_depth=7, min_samples_leaf=4, min_samples_split=10,random_state = 42),
                        
                       'kneighbour':KNeighborsRegressor(n_neighbors= 5,weights = 'uniform',algorithm= 'auto', leaf_size = 30, p = 2) }



## For dataset-2 Hyperparameters
Algorithms_regression_2 ={'Linear' : LinearRegression(fit_intercept=True), 
                        
                        'random_forest' : RandomForestRegressor(n_estimators= 100,max_depth = None,random_state=42),
                        
                        'decision_tree' : DecisionTreeRegressor(max_features='log2', min_samples_leaf=6,
                         min_samples_split=10,random_state=42),
                        
                        'xgboost':XGBRegressor(objective ='reg:squarederror',random_state = 42) ,
                        
                        'gradient_boost': GradientBoostingRegressor(n_estimators = 100,learning_rate = 0.1,max_depth=7, min_samples_leaf=4, min_samples_split=2,random_state = 42),
                        
                       'kneighbour':KNeighborsRegressor(n_neighbors= 5,weights = 'distance',algorithm= 'auto', leaf_size = 30, p = 1) }




## For dataset-3 Hyperparameters
Algorithms_regression_3 ={'Linear' : LinearRegression(fit_intercept=True), 
                        
                        'random_forest' : RandomForestRegressor(n_estimators= 50,max_depth = 5,random_state=42),
                        
                        'decision_tree' : DecisionTreeRegressor(max_features='log2', max_depth = 3,min_samples_leaf=1,
                         min_samples_split=5,random_state=42),
                        
                        'xgboost':XGBRegressor(objective ='reg:squarederror',random_state = 42) ,
                        
                        'gradient_boost': GradientBoostingRegressor(n_estimators = 50,learning_rate = 0.1,max_depth=3, min_samples_leaf=2, min_samples_split=10,random_state = 42),
                        
                       'kneighbour':KNeighborsRegressor(n_neighbors= 5,weights = 'distance',algorithm= 'brute', leaf_size = 30, p = 2) }                        





####Training LOOP
######Change also the name of dictionary as specified abovee as 
# Algorithms_regression_1, - Dataset_1 hyperparameters
# Algorithms_regression_2, - Dataset_2 hyperparameters
# Algorithms_regression_3, - Dataset_3 hyperparameters

for names,models in Algorithms_regression_1.items(): 
  
    
    models.fit(X_train_scaled, y_train) 
    
    y_train_prediction = models.predict(X_train_scaled)
    
    predicts = models.predict(X_test_scaled)         
    
    #scatter_plots(y_test,predicts,names)
    #residual_plots(y_test,predicts,names)
    
    results_storing_training[names]= {'RMSE':np.sqrt(mean_squared_error(y_train, y_train_prediction)), 'R2': r2_score(y_train, y_train_prediction),'mean_error':mean_absolute_error(y_train, y_train_prediction)}
    
    results_storing[names]= {'RMSE':np.sqrt(mean_squared_error(y_test, predicts)), 'R2': r2_score(y_test, predicts),'mean_error':mean_absolute_error(y_test, predicts)}

    print(models)
    
     
### Metrics storing in the dictionaries while training
for model_training, metrics_training in results_storing_training.items():
    
    print(f"\n{model_training} Results for regression during training:")
    
    print(f"RMSE_during_Training: {metrics_training['RMSE']}")
    
    print(f"R-squared_during_training: {metrics_training['R2']}")
    
    print(f"mean_abs_during_training: {metrics_training['mean_error']}")   



### Metrics storing in the dictionaries while testing
for model_test, metrics_test in results_storing.items():
    
    print(f"\n{model_test} Regression_Results:")
    
    print(f"RMSE: {metrics_test['RMSE']}")
    
    print(f"R-squared: {metrics_test['R2']}")
    
    print(f"mean_abs: {metrics_test['mean_error']}")       




                       
###### REGRESSION INTO CLASSIFICATION PROBLEM
''' 1. Then convert the regression into classification problem and apply all classifiers algorithM
    2. RMSE, MAE and R2 score interpretable as target values after inverse transformation
    3. Dictionary initialisation for classification algorithms the datas to be trained and tested on. '''
    
algorithms_for_classification = {
                        'random_classi' : RandomForestClassifier(random_state=42),
                        
                        'decision_classi' : DecisionTreeClassifier(random_state=42),
                        
                        'neighbour ' : KNeighborsClassifier(),
                        
                    'boostings ': GradientBoostingClassifier(random_state = 42),
                   
                    'xgbclassifier':XGBClassifier(random_state = 42)}







###### Function call of the data preprocessing extracting the X,y values from the dataset
#### Big dataset Frames Per Seconds
X_classification_train,y_classification_train = dataset_1(path_4)
print('FOR CLASSIFICATION_X_VALUES ',X_classification_train)
print('FOR CLASSIFICATION_Y_VALUES ',y_classification_train)


##### California Housing dataset - Dataset 2

# X_classification_train,y_classification_train = dataset_2(Housing)
# print('FOR CLASSIFICATION_X_VALUES ',X_classification_train)
# print('FOR CLASSIFICATION_Y_VALUES ',y_classification_train)


##### Diabetes dataset - Dataset 3

# X_classification_train,y_classification_train = dataset_3(dia)
# print('FOR CLASSIFICATION_X_VALUES ',X_classification_train)
# print('FOR CLASSIFICATION_Y_VALUES ',y_classification_train)




def uni_discretizer(n_bins,encode,strategy):                    #### Function initalisation for changing the regression into classification problem for training and testing
    
    return KBinsDiscretizer(n_bins = n_bins,encode = encode,strategy  = strategy,subsample = None ,random_state = 42)


list_of_approaches = [ 'kmeans','uniform', 'quantile']          ######Initialisation of the list with different strategies of the sci-kit learn for data conversion

for approaches in list_of_approaches:
    
    print(f'Starting for the approach {approaches}')
    
    '''1. initialisation of the ranges for number of bins parameter to be added to discretization function
       2. When given 4 it shows three number of bins so, bins are constructed as (bins_parameter-1) number of classes for data discretization
       3. Step size of 2 is defined in the ranges this will give [3,5,7] '''
    for bins_parameter in range (4,10,2): 
                          
        print(f'Starting for the approach {approaches} with number of bins {bins_parameter-1}') 
        
        
        ###Function call for discretization with different bins and strategies.
        y_y = uni_discretizer(bins_parameter,'ordinal',approaches)
        y_transform_regression = y_y.fit_transform(y_classification_train.reshape(-1,1)) ##
        
        ##Checking the bin edges to check the edges of the transformed datas.
        print(f'BIN_EDGES_OF_TRANSFORMED_DATA_for_target_variable_for_{bins_parameter-1}_number of bins_with_approach_{approaches} : ',y_y.bin_edges_[0],y_y.bin_edges_)
        
        

        y_trans = y_transform_regression.reshape(-1,)
        print('Y_TRANSFORMATION:',y_trans.shape)
        print(y_transform_regression.shape)
        
        
        ### After transformstion of the data train and split the x,y values before training and prediction of the models for classification task.
        X_train_class, X_test_class, y_train_class, y_test_class= train_test_split(X_classification_train, y_trans ,train_size = 0.75, test_size=0.25, random_state=42) ##,shuffle=False
        print('Classification Data Shapes :',X_train_class.shape, X_test_class.shape, y_train_class.shape,y_test_class.shape)
        
        ### Preprocessing the data befroe training and testing.
        X_train_scaled_class = scaler.fit_transform(X_train_class)
        X_test_scaled_class = scaler.fit_transform(X_test_class)
        print(X_test_scaled)
        
        
        
        
        ### Checks how the datas are splitted - Uncomment it to check and save (Change the directory path for saving the plots)
        
        # # plt.figure(figsize=(10,5))
        # # plt.scatter(y_test,y_test_class)
        # # plt.title(f'Data divided in different classes using ordinal strategy and number of bins_{approaches}_and_{bins_parameter-1}')
        # # plt.ylabel('Ordinal Strategy')
        # # plt.xlabel('Frames Per Second - Target Variable')
        # # plt.savefig(f"C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\data_plots\\data_divison_visualisation_{approaches}_and_{bins_parameter-1}.png")
        # # plt.grid(True)
        
        # # plt.figure(figsize=(10,5))
        # # plt.scatter(y_train_class,y_train)
        # # plt.title(f'Train Data divided in different classes using ordinal strategy and number of bins_{i}_and_{t}')
        # # plt.xlabel('Ordinal Strategy')
        # # plt.ylabel('Frames Per Second - Target Variable')
        # # plt.savefig(f"C:\\Users\\Shah\\Desktop\\Studienarbeit\\scatter_plot_reslt\\data_plots\\Train_Data\\data_divison_visualisation_{approaches}_and_{bins_parameter-1}.png")
        # # plt.grid(True)
       
        
        ### Train and test the different models for classification and check the metrics to be evaluated
        for classifier_names, diff_classification_algo in algorithms_for_classification.items():
    
            ###Model fitting o the different algorithms of the classification domain
            diff_classification_algo.fit(X_train_scaled_class,y_train_class)
    
            
            ###prediction of the different algorithms on the test dataset after training
            prediction_classification = diff_classification_algo.predict(X_test_scaled_class)

            
            ###Inverse transformation of the predicition for comparing the y_test data for the regression
            inversion = y_y.inverse_transform(prediction_classification.reshape(-1,1)) 
    
            ###Data plots of the scatter type for original regression test data with comparision to inversed transform datas
            scatter_plots_inverse(y_test,inversion,classifier_names,approaches,bins_parameter-1)
            
           
            #### Metrics to be evaluated after the inversion wqith the original test dataset splitted during regression from the 
            #### function definition of spliting_datas_regression(X,y)
            root_mean = np.sqrt(mean_squared_error(y_test,inversion))                           ### root mean sqaure error

            mean_abs_check = mean_absolute_error(y_test,inversion)                              ### mean absolute error
            
            accuracies_check_3 = abs(r2_score(y_test,inversion))                                ### R2 score
            
           
            

            ### Check the classification metrics after the conversion from the regression to classification
            accuracies_check = accuracy_score(y_test_class,prediction_classification)           ## Classification accuracy
            
            f1_scorings = f1_score(y_test_class,prediction_classification,average = 'weighted') ## F1 score
            
            print('CLASSIFICATION ACCURACY CHECK After reg to class transformed_ACCURACY_SCORE : ',accuracies_check,approaches,bins_parameter-1)
            
            print('CLASSIFICATION ACCURACY CHECK After reg to class transformed_F1_SCORE : ',f1_scorings,approaches,bins_parameter-1)
           


            #### Printing and checking the values of the metrics after inversing the classification to  Regression domain
            print('Root_MEAN_squared for classification to regression after inverse tranformating - ',root_mean)
    
            print('MEAN_abs_check for classification to regression after inverse tranformating - ',  mean_abs_check)
            
            print('Accuracy for classification to regression after inverse tranformating ',accuracies_check_3)
            
            
            ### Completion the training and testing on a particular model
            print(f'The model has beend trained on - {diff_classification_algo}') 
            

         
    ### Completion of the training and testing on a particular model with approaches and (bins parameters - 1)    
    print(f'completed training and testing for the approach {approaches} with {bins_parameter-1}')





   
 



