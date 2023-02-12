"IMPORTING LIBRARIES"
import numpy as np 
import os
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import logging
import openml
import argparse
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
import os.path 
from configs import config
import typing

warnings.filterwarnings("ignore")


'''Collecting necessary inputs using argparse'''
def parse_args():
    all_regressors = ['decision_tree','random_forest','k_nearest_neighbors']
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True, 
                        help='the openml task id')
    parser.add_argument('--openml_server', type=str, default=None, 
                        help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None,
                        help='the apikey to authenticate to OpenML')
    parser.add_argument('--regressor_name', type=str, choices=all_regressors,
                        default='decision_tree',
                        help='the regressor to run')
    parser.add_argument('--upload_result', action='store_true',
                        help='if true, results will be immediately uploaded to OpenML.'
                             'Otherwise they will be stored on disk. ')
    parser.add_argument('--run_defaults', action='store_true',
                        help='if true, will run default configuration')
    return parser.parse_args()

'''To delete run files from local directory'''
def del_outputdir():
    file_name = 'myrun'
    path=(os.path.abspath(file_name))

    if os.path.exists(path + "\\flow.xml"):
        os.remove(path + "\\flow.xml")   
    if os.path.exists(path + "\\model.pkl"):
        os.remove(path + "\\model.pkl")
    if os.path.exists(path + "\\predictions.arff"):
        os.remove(path + "\\predictions.arff")
    if os.path.exists(path + "\\description.xml"):
        os.remove(path + "\\description.xml")
  
    else:
            print("The system cannot find the file specified")
       
'''Numerical data and Catogarical data preprocessing'''

args = parse_args()
task=openml.tasks.get_task(args.task_id)
numeric_indices= typing.List[int]
nominal_indices= typing.List[int]
nominal_indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
numeric_indices = task.get_dataset().get_features_by_type('numeric', [task.target_name])

numeric_transformer = sklearn.pipeline.make_pipeline(
    sklearn.impute.SimpleImputer(strategy='mean'),
    sklearn.preprocessing.StandardScaler())

categorical_transformer = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))

col_trans = sklearn.compose.ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_indices),
        ('nominal', categorical_transformer, nominal_indices)],
    remainder='drop')


'''Building a pipeline'''
def pipeline():
    args = parse_args()
    reg_name= args.regressor_name
    reg=config(reg_name)
    
    reg_pipeline = sklearn.pipeline.make_pipeline(col_trans,
                                             reg)
    return reg_pipeline



def runbot():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    args = parse_args()
    if args.openml_apikey:
        openml.config.apikey = args.openml_apikey
    if args.openml_server:
        openml.config.server = args.openml_server
    else:
        openml.config.server = 'https://test.openml.org/api/v1/'
    reg_pipeline=pipeline()
    run = openml.runs.run_model_on_task(reg_pipeline, args.task_id)
    score = []
    evaluations = run.fold_evaluations['mean_absolute_error'][0]
    for key in evaluations:
        score.append(evaluations[key])
    mae=np.mean(score)
    print('mean_absolute_error:', mae)
    print('mean_squared_error:' ,np.mean(run.get_metric_fn(sklearn.metrics.mean_squared_error)))
    print('r2_score:' ,np.mean(run.get_metric_fn(sklearn.metrics.r2_score)))
    logging.info('Task %d - %s; Accuracy: %0.2f' % (args.task_id, task.get_dataset().name, mae))
    
    del_outputdir()
    run.to_filesystem(directory= 'myrun')
    print('upload_result=',args.upload_result)
    
    if args.upload_result:
        run.publish()
        print(run)
        print('Results Uploaded to Openml and stored in local folder myrun')
    else:
        print('Results stored in local folder myrun')

if __name__ == '__main__':
    runbot()
    
'''   

from ConfigSpace import ConfigurationSpace
from ConfigSpace import *
import sklearn


cs = ConfigurationSpace()

criterion = CategoricalHyperparameter(
    "criterion", ["mse", "friedman_mse", "mae"]
 )
max_features = Constant("max_features", 1.0)
max_depth_factor = UniformFloatHyperparameter(
    "max_depth_factor", 0.0, 2.0, default_value=0.5
)
min_samples_split = UniformIntegerHyperparameter(
    "min_samples_split", 2, 20, default_value=2
)
min_samples_leaf = UniformIntegerHyperparameter(
    "min_samples_leaf", 1, 20, default_value=1
)
min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
min_impurity_decrease = UnParametrizedHyperparameter(
    "min_impurity_decrease", 0.0
)

cs.add_hyperparameters(
     [
         criterion,
         max_features,
         #max_depth_factor,
         min_samples_split,
         min_samples_leaf,
         min_weight_fraction_leaf,
         max_leaf_nodes,
         min_impurity_decrease,
     ]
 )

model= sklearn.tree.DecisionTreeRegressor()
config = cs.sample_configuration()
params = config.get_dictionary()
model.set_params(**params)
print(model)
#run = openml.runs.run_model_on_task(model=task_id,task=clf_pipeline)


#openml.config.apikey ='2f81281c6763a4065d7a611b865bb9d6'
#task_id=2295
#n=5
#task = openml.tasks.get_task(task_id)
#df = task.get_dataset()
#X, y, _, _ = df.get_data(target=df.default_target_attribute);
#attribute_names = list(X)
#print(task.dataset_id)
#print(task.task_type)
#print(task.target_name)
#print(task.estimation_procedure_id)
#print(task.evaluation_measure)
#print(task.estimation_parameters)
#print(task.task_type_id)

openml.config.apikey ='2f81281c6763a4065d7a611b865bb9d6'
task_id=4823
task = openml.tasks.get_task(task_id)
df = task.get_dataset()
X, y, _, _ = df.get_data(target=df.default_target_attribute);
attribute_names = list(X)
print(task.dataset_id)
#print(task.flow_id)
print(task.task_type)
print(task.target_name)
print(task.estimation_procedure_id)
print(task.evaluation_measure)
#print(task.kwargs)
#print(task.estimation_procedure_type)
print(task.estimation_parameters)
#print(task.data_splits_url)
print(task.task_type_id)


openml.tasks.OpenMLRegressionTask(task_type_id= task.task_type_id, task_type= task.task_type, data_set_id= task.dataset_id, target_name=task.target_name, estimation_procedure_id = task.estimation_procedure_id, estimation_parameters= task.estimation_parameters, task_id=task_id , evaluation_measure=task.evaluation_measure)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean'))
])
cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot',OneHotEncoder(categories='auto',handle_unknown='ignore', sparse=False))
])


from sklearn.compose import ColumnTransformer

col_trans = ColumnTransformer(transformers=[
    ('num_pipeline',num_pipeline,num_cols),
    ('cat_pipeline',cat_pipeline,cat_cols)
    ],
    remainder='drop',
    n_jobs=-1)


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0)
clf_pipeline = Pipeline(steps=[
    ('col_trans', col_trans),
    #('my_pipeline', my_pipeline),
    ('model', clf)
])
#clf_pipeline.fit()

for i in range(args.n_executions):
    run = openml.runs.run_model_on_task(clf_pipeline, task_id)'''
