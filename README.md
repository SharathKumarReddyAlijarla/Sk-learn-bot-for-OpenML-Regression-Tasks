# SA_openml_bot_reg
## README file for Bot code
### Introduction
This code performs various operations for building a machine learning pipeline and evaluating its performance on a given OpenML task. The code uses various libraries such as numpy, sklearn, and openml to accomplish this. The code includes the following functionalities:\
•	Argparse for collecting necessary inputs for running the model.\
•	Deleting run files from the local directory\
•	Preprocessing numerical and categorical data using the ColumnTransformer of the scikit-learn library.\
•	Hyperparameter Tuning using the config from configs.py file\
•	Building a pipeline using the make_pipeline method of the scikit-learn library.\
•	Evaluating the performance of the model on a given OpenML task and uploading the results to the OpenML server
### Requirements

•	**numpy**                     1.21.5\
•	**openml**                    0.12.2\
•	**scikit-learn**              0.21.0\
•	**configspace**               0.4.19


### Usage
To use the code, you need to provide the following inputs via command line arguments:

•	**task_id:** the OpenML task id\
•	**openml_server:** the OpenML server location (optional)\
•	**openml_apikey:** the API key to authenticate to OpenML (optional)\
•	**regressor_name:** the name of the regressor to run (default is decision_tree)\
•	**upload_result:** a flag to indicate if the results should be immediately uploaded to OpenML or stored on disk.\
•	**run_defaults:** a flag to indicate if the code should run with default configuration.

### Code Structure
The code starts by importing necessary libraries and setting the logging level. The parse_args function collects the inputs via argparse. The del_outputdir function deletes the run files from the local directory.

The preprocessing of numerical and categorical data is performed using the ColumnTransformer of the scikit-learn library. The pipeline function builds the pipeline by combining the preprocessor and the regressor.

The runbot function is the main function of the code that uses the run_model_on_task method of the OpenML library to evaluate the performance of the model on a given task and uploads the results to the OpenML server if the upload_result flag is set.

### Conclusion
This code provides a comprehensive solution for building a machine learning pipeline and evaluating its performance on a given OpenML task. The code takes care of preprocessing numerical and categorical data, building a pipeline, and uploading the results to the OpenML server.




