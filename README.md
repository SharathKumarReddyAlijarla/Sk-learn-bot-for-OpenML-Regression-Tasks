# SA_openml_bot_reg
## README file for Bot code
### Introduction
Scikit-learn bot that can be used to automatically run scikit-learn Regressors on [OpenML tasks](https://www.openml.org/search?type=task). This code performs various operations for building a machine learning pipeline and evaluating its performance on a given OpenML task. The code uses various libraries such as numpy, sklearn, and openml to accomplish this. The code includes the following functionalities:\
•	Argparse for collecting necessary inputs for running the model.\
•	Deleting run files from the local directory [myrun](https://github.com/SharathKumarReddyAlijarla/Sk-learn-bot-for-OpenML-Regression-Tasks/tree/main/myrun).\
•	Preprocessing numerical and categorical data using the ColumnTransformer of the scikit-learn library.\
•	Hyperparameter Tuning using the config from [configs.py](https://github.com/SharathKumarReddyAlijarla/Sk-learn-bot-for-OpenML-Regression-Tasks/blob/main/configs.py) file.\
•	Building a pipeline using the make_pipeline method of the scikit-learn library.\
•	Evaluating the performance of the model on a given OpenML task and uploading the results to the [OpenML](https://www.openml.org/) server.

### Requirements
•	**[NumPy](https://numpy.org/)**                     1.21.5\
•	**[OpenML-Python](https://pypi.org/project/openml/)**             0.12.2\
•	**[Scikit-learn](https://pypi.org/project/scikit-learn/)**              0.21.0\
•	**[ConfigSpace](https://pypi.org/project/ConfigSpace/)**               0.4.19

### Usage
To use the code, you need to provide the following inputs via command line arguments:\
•	**task_id:** the OpenML task id.\
•	**openml_server:** the [OpenML](https://www.openml.org/) server location. \
•	**openml_apikey:** the [API key](https://new.openml.org/auth/api-key) to authenticate to OpenML. \
•	**regressor_name:** the name of the regressor to run (default is decision_tree).\
•	**upload_result:** a flag to indicate if the results should be immediately uploaded to OpenML or stored in Local directory.\
•	**run_defaults:** a flag to indicate if the code should run with default configuration.


<img width="500" alt="keys" src="https://user-images.githubusercontent.com/122915971/220946826-fd7acc7f-0709-454f-9a94-c113d602cbf2.png"> 


### Code Structure
The code starts by importing necessary libraries and setting the logging level. The parse_args function collects the inputs via argparse. The del_outputdir function deletes the run files from the local directory.

The preprocessing of numerical and categorical data is performed using the ColumnTransformer of the scikit-learn library. The pipeline function builds the pipeline by combining the preprocessor and the regressor.

The runbot function is the main function of the code that uses the run_model_on_task method of the OpenML library to evaluate the performance of the model on a given task and uploads the results to the OpenML server if the upload_result flag is set.

### Obtaining results
Usually, running the sklearn-bot is done so that the results can be uploaded to OpenML. Once the results have been stored on OpenML, we can view them anytime in [OpenML Runs](https://www.openml.org/search?type=run&sort=date).

<img width="578" alt="runanalysis" src="https://user-images.githubusercontent.com/122915971/220950229-ec0cb949-dc81-4bae-a0d9-0ccc14a74a38.png">


### Future Work and Limitations
Future work on a scikit-learn bot for OpenML regression tasks could include:\
•	Improved Model Selection: The bot can be enhanced to undertake more advanced model selection, such as using ensemble methods or systematically comparing several models at the same time.\
•	Integration with Other Technologies: In order to provide a more comprehensive solution for regression tasks, the bot may be integrated with other machine learning technologies such as TensorFlow or PyTorch.\
\
Despite these enhancements, the scikit-learn for OpenML regression tasks bot still has certain limitations:\
•	Few regression Tasks only: The bot can only be used for few regression tasks at the moment and also cannot be utilized for tasks like classification or clustering.\
•	The bot only supports scikit-learn currently and does not support any other machine learning libraries.\
•	Limited to OpenML: The bot only supports OpenML tasks and data at this time, and it does not support any other platforms or data sources.

### Conclusion
This code provides a comprehensive solution for building a machine learning pipeline and evaluating its performance on a given OpenML task. The code takes care of preprocessing numerical and categorical data, building a pipeline, and uploading the results to the OpenML server. In conclusion, a potent tool for automating machine learning experiments is the scikit-learn bot created for regression tasks on OpenML. The bot can run several regression techniques, assess their effectiveness, and publish the results to the server by utilizing the scikit-learn package and OpenML platform. The bot may explore a wide range of hyperparameter settings thanks to the use of suitable configuration spaces, which guarantees this and gives useful insights into the most effective algorithms and hyperparameters for regression tasks. All things considered, the scikit-learn bot is a useful tool for anyone trying to speed up their machine learning testing process and enhance their output.

### References
1.	Joaquin Vanschoren and Jan N. van Rijn and Bernd Bischl and Luis Torgo. OpenML: networked science in machine learning.SIGKDD Explorations 15(2), pp 49-60, 2013
2.	Matthias Feurer and Jan N. van Rijn and Arlind Kadra and Pieter Gijsbers and Neeratyoy Mallik and Sahithya Ravi and Andreas Mueller and Joaquin Vanschoren and Frank Hutter. OpenML-Python: an extensible Python API for OpenML.arXiv 1911.024902020
3.	Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
4.	Buitinck, L., Louppe, G., Blondel, M., Pedregosa, F., Mueller, A., Grisel, O., ... Varoquaux, G. (2013). API design for machine learning software: experiences from the scikit-learn project. In ECML PKDD Workshop: Languages for Data Mining and Machine Learning (pp. 108-122).
5.	M. Feurer, J. T. Springenberg, and F. Hutter. Initializing bayesian hyperparameter optimization via meta-learning. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pages 1128–1135. AAAI Press, 2015.
6.	Indauer, M., Eggensperger, K., Feurer, M., Biedenkapp, A., Marben, J., Müller, P., & Hutter, F. (2019). BOAH: A Tool Suite for Multi-Fidelity Bayesian Optimization & Analysis of Hyperparameters. arXiv:1908.06756 [cs.LG].
7.	Python Software Foundation. argparse - Command line option and argument parsing module. [Online]. Available: https://docs.python.org/3/library/argparse.html [Accessed:  Feb 2023].
8.	OpenML. OpenML/sklearn-bot: A sklearn bot for OpenML. [Online]. Available: https://github.com/openml/sklearn-bot [Accessed: Feb 2023].
9.	Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
10.	Feurer, M., Eggensperger, K., Falkner, S., Lindauer, M., & Hutter, F. (2020). Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning. arXiv preprint arXiv:2007.04074 [cs.LG].





