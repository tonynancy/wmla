
## Introduction to Watson Machine Learning Accelerator

Watson Machine Learning Accelerator is a deep learning service for data scientists on Cloud Pak for Data.

Watson Machine Learning Accelerator can be connected to Watson Machine Learning to take advantage of the multi-tenant resource plans that manage resource sharing across Watson Machine Learning projects. With this integration, data scientists can use the Watson Machine Learning Experiment Builder and Watson Machine Learning Accelerator hyperparameter optimization.
Watson Machine Learning Accelerator provides the following benefits on Cloud Pak for Data:

    Distributed deep learning architecture and hyper-parameter search and optimization that simplifies the process of training deep learning models across a cluster for faster time to results.
    Advanced Kubernetes scheduling including consumer resource plans, ability to run parallel jobs and dynamically allocate GPU resources.
    Included ready-to-use deep learning frameworks, such as TensorFlow and PyTorch.

## Supported deep learning frameworks

Watson Machine Learning Accelerator support the following deep learning frameworks and its version. They are TensorFlow 2.14.1, PyTorch 2.1.2, Tensorboard 2.12.2, torchVision 0.15.2, OpenCV 4.7.0, scikit-learn 1.1.3, XGBoost 1.7.6, ONNX 1.13.1, PyArrow 11.0.0, Python	3.10.12. PyTorch in WMLA supports single node training, distributed training, and elastic distributed training. TensorFlow in WMLA supports single node training, distributed training, and elastic distributed training (with TF.Keras).

## Features of Watson Machine Learning Accelerator
Watson Machine Learning Accelerator includes key features like elastic distributed training, inference, model training and hyperparameter optimization and tuning.

### Elastic distributed training
Watson Machine Learning Accelerator utilizes elastic distributed training as part of its general distribution deep learning framework.

Many of the recent deep learning applications involve large training models and large volumes of training data which can consume large amounts of processing time. With the development of hardware technology, GPU devices speed up the process of training models. As a result, a distributed deep learning framework leveraging the GPU devices of a cluster and implementing optimal distribution training algorithms are necessary.

Watson Machine Learning Accelerator provides comprehensive support for distributed deep learning across multi-GPU and multi-node clusters which is compatible with existing PyTorch models. Elastic distributed training supports various parallelization schemes which have a gradient exchange descent greater than 80% speedup ratio. It is also capable to efficiently scale out on a Watson Machine Learning Accelerator cluster.

The elastic distributed training engine uses a fine grained control for training. This is an alternate to the coarse grained strategy, where the resource utilization is capped and no additional GPUs can be added to the training phase. With elastic distributed training, the scheduler distributes the task and the session scheduler dynamically issues resources from the resource manager, allowing for additional GPUs to be added.
In addition, the following functions are supported with the elastic distributed training engine:

    Single-node, multi-GPU training support without parameter server .
    Multi-node, multi-GPU training support with parameter server.
    Both synchronous and asynchronous distributed training algorithms support for multi-node, including synchronous gradient data control algorithm, and asynchronous gradient data control algorithm.
    NVIDIA Collective Communications Library (NCCL) support for broadcasting, reducing gradient and weight data across multiple GPUs.

### Hyperparameter turning and optimization
Watson Machine Learning Accelerator features hyperparameter tuning and optimization.

Hyperparameters are parameters whose values are set before starting the model training process. Deep learning models, including convolutional neural network (CNN) and recurrent neural network (RNN) models can have anywhere from a few hyperparameters to a few hundred hyperparameters. The values specified for these hyperparameters can impact the model learning rate and other regulations during the training process as well as final model performance.

Watson Machine Learning Accelerator uses hyperparameter optimization algorithms to automatically optimize models. The algorithms used include Random Search, Tree-structured Parzen Estimator (TPE) and Bayesian optimization based on the Gaussian process. These algorithms are combined with a distributed training engine for quick parallel searching of the optimal hyperparameter values. 

### Using framework via the Watson Machine Learning Accelerator command line interface (CLI)

Use any deep learning framework with Watson Machine Learning Accelerator using the dlicmd tool.

Frameworks that are installed using the dlicmd tool can be run from a command line interface. These frameworks are isntalled and utilize single node training on the Watson Machine Learning Accelerator cluster. Any framework can be added using the dlicmd tool and used with cluster resources to run deep learning tasks.

### Elastic distributed training
Use this document to learn how to run elastic distributed training workloads using GPU devices. GPU devices are dynamically allocated to the training job without stopping training. 

#### Update model files
To run elastic distributed training, update your training model files to make the following two changes:
  - Create a FabricModel instance to define an elastic distributed training model, see https://www.ibm.com/docs/en/wmla/5.0?topic=training-elastic-distributed#wmla_workloads_elastic_distributed_training__edt-fabricmodel-definition for details on defining a FabricModel.
  - To start training a model, run the train function of FabricModel, see https://www.ibm.com/docs/en/wmla/5.0?topic=training-elastic-distributed#wmla_workloads_elastic_distributed_training__edt-fabricmodel-methods for details.

#### Get started
With all the required changes to your model code, a simple training entry code can be as below:
```
from fabric_model import FabricModel

def get_dataset():
    # Prepare or clean data
    return train_dataset, test_dataset

model = ...
optimizer = ...
loss_function = ...

edt_model = FabricModel(model, get_dataset, loss_function, optimizer)

epochs = ...
batch_size = ...
engines_number = ...
edt_model.train(epochs, batch_size, engines_number)
```
Define dataset function
```
def get_dataset():
    # Prepare or clean data
    return train_dataset, test_dataset

```

Both train and test dataset should be a map-style dataset, for PyTorch, it can be an instance of torch.utils.data.Dataset. for TensorFlow, it will be similar format as PyTorch map-style dataset with function __getitem__ and __len__ defined like below.
```
class EDTTensorFlowDataset:
    def __init__(self, x, y) -> None:

      self.x = np.array(x)
      self.y = np.array(y)

    def __getitem__(self, index):
        """Gets sample at position `index`.
        Args:
            index: position of the sample in data set.
        Returns:
            tuple (x, y)
            x - feature
            y - label
            x and y can be scalar, numpy array or a dict mapping names to the corresponding array or scalar.
        """
        return self.x[index], self.y[index]

    def __len__(self):
        """Number of samples in the dataset.
        Returns:
            The number of samples in the dataset.
        """
        return len(self.x)

```

Define custom logger callback
A custom logger can be defined as below and it can run on either as driver logger callback or worker logger callback. If a custom logger callback is used, the default logger callback is replaced with the custom logger callback.

```
class MyLoggerCallback():
    '''
    Abstract base class used to build new logger callbacks.
    '''

    def log_train_metrics(self, metrics, iteration, epoch, workers):
        '''
        Log metrics after training a batch.

        Parameters:
            metrics (dict): dictionary mapping metric names (strings) to their values. On driver, it will be the average accumulated
                            metric values among all batchs in current training epoch. On worker, it will be the on step training metrics.
            iteration (int): current iteration number. On driver, it will be the total number of iterations across all workers that have 
                             already been run. On worker, it will be the total number of iterations the particular worker has already run.
            epoch (int): current epoch number.
            workers (int): number of training workers or GPUs
        '''

    def log_test_metrics(self, metrics, iteration, epoch):
        '''
        Log metrics after training a batch.

        Parameters:
            metrics (dict): dictionary mapping metric names (strings) to their values. On driver, it will be the average accumulated
                            metric values among all test data. On worker, it will be the on step test metrics.
            iteration (int): current iteration number. On driver, it will always be 0. On worker, it will be the total number of 
                             test iterations the particular worker has already run.
            epoch (int): current epoch number.
        '''

    def on_train_end(self):
        '''
        Log metrics when training is finished.
        '''

    def on_train_begin(self):
        '''
        Log metrics when training is started.
        '''
```

Define a custom batch collation function
A custom batch collation function can be used to define how to combine samples from original dataset generated by the dataset function
```
def batch_collate_function(batch)
    '''
    Parameters
    batch: a list of training samples, each sample is a tuple with one features value and one label value in it, basically what the __getitem__ of the dataset instance returns.
    Return:
A tuple (a batch of data features, a batch of data labels)
'''
```

#### Running elastic distributed training
There is not much difference to launch elastic distributed training compared with other regular training on Watson Machine Learning Accelerator, except it requires some particular values for training interfaces.
  - Using data connection through REST API
  - POST request to REST API /platform/rest/deeplearning/v1/execs to start training workload.
  - Using data connection through the command line interface (CLI)
Use the Watson Machine Learning Accelerator CLI. You can download the CLI from the Watson Machine Learning Accelerator console, see: https://www.ibm.com/docs/en/SSFHA8_5.0/cm/dlicmd.html.
These are the primary parameters that could make a difference for elastic distributed training compared with other regular training on Watson Machine Learning Accelerator.

### Inference
Using the elastic distributed inference capability in WML Accelerator, users can create inference services and deploy published models.

#### Download and configure the elastic distributed inference (dlim) command line utility 
To download and configure an inference service using the following step,
  - step1, To use the dlim command, you must to download and obtain the dlim command line utility for Watson Machine Learning Accelerator from IBM Git at https://github.com/IBM/wmla-assets/
  - step2, Add dlim to PATH, for example:PATH=$PATH:/usr/local/bin/dlim
  - step3, Configure dlim: dlim config -c https://cpd_route
  - step4, To run dlim commands, use a prefetched JWT token directly, for example: dlim config -t -u username -k apikey
  - step5, To see what subcommands are available with the dlim command, run the dlim --help command: dlim --help
  - Examples, PATH=$PATH:/usr/local/bin/dlim, dlim config -c https://cpd-cpd-instance.apps.ibm.com, dlim config -t -u username -k apikey, dlim model list
  - Note,If you are running dlim in a Watson Studio notebook, dlim cannot be configured.  To run dlim commands, use a prefetched JWT token directly, For example: dlim model list --jwt-token $USER_ACCESS_TOKEN --rest-server cpd_route. To use a prefetched token, create a local user that contains your Cloud Pak for Data username and API, for example: dlim config -t -u username -k apikey.

#### Create an inference service
Before you begin, make sure that you have the model files that you are creating the inference service for. To create the inference service, you will need to create a kernel file, a model.json file, a readme file, and the model to be deployed.
To create an inference service using the following step,
  - step1, Create a working directory and copy your model files into that directory.
  - step2, Create the model.json file for the model and add it to the directory. 
  - step3, Create a README.md file. A readme file describes what the input data looks like and what type of inference server response is expected. Add this file to your working directory.
  - step4, Create a kernel file, see Create a kernel file for an inference service. The kernel file must be added to your working directory.
  - step5, Deploy the model: dlim model deploy -p working_directory.
  - Note: Elastic distributed inference creates a model service deployment, and the cpd-scheduler allocates memory to run the service deployment. There must be enough GPUs to run the model service for the kernel pods to start. 
  
#### Start an inference service
Before starting the service, you must update the kernel profile in order for the model information to appear in the Profile tab of your deployment. Use the dlim command to update your kernel profile: dlim model updateprofile model-name -f profile.json.
To start an inference service using the following step,
  - step1, Ensure that you have downloaded and configured the dlim utility tool, see https://www.ibm.com/docs/en/SSFHA8_5.0/us/edi-cli.html.
  - step2, Start inferenc service, dlim model start model-name.
  - step3, Verify that the model was started: dlim model view model-name -s.

#### Stop an inference service
Ensure that you have downloaded and configured the dlim utility tool
To stop an inference service using the following step,
  - Run the following command:dlim model stop model-name. where model-name is the name of the model that you want to stop. 

#### Edit an inference service
To edit an inference service using the following step,
  - step1, Before editing the deployed model, stop the inference service, dlim model stop model_name
  - step2, Get the model json file, dlim model viewprofile model_name -j > modelA_profile.json
  - step3, Edit the JSON file and save your changes.The following parameters are available:
    - Replica: Number of copies of the service to run for a model.
    - Kernel delay release time: Time (in seconds) to wait after the system detects the number of kernels that are running is higher than the load requires before the       - system stops the extra kernels. Must be greater than 0.
    - Kernel Min: Minimum number of kernels to always keep running. Must be greater than 0. Do not set higher than the number of slot resources you want to consume all the time, and do not set higher than the total number of slot resources in the resource plan.
    - Kernel Max: Maximum number of kernels to scale up to. Specify -1 for unlimited number of kernels or a number greater than or equal to the value set for Kernel Min.
    - Schedule interval: How often the service re-evaluates the number of kernels that are running based on current load. Time in seconds. Must be greater than 0.
    - Stream discard slow tasks: If true, after new tasks are completed, older tasks are discarded instead of being returned to the gRPC streaming client. Applies to gRPC streaming clients only, not applicable to REST clients. To enabled, specify true, otherwise, set to false.
    - Stream number per group: Number of streams per resource group.
    - Task Execution Timeout: Time (in seconds) to wait for an individual inference request task to complete. Specify -1 for no timeout or a value greater than 0.
    - Task Batch Size: Number of tasks that can be dispatched to a single kernel in a single call. Used in a GPU enabled kernel that uses a high batch size to reach optimal performance. Must be greater than 0.
    - Connection timeout: Time (in seconds) that the service waits for a kernel to report that it started. Must be greater than 0.
    - Namespace: Namespace that the kernel is running in. This value cannot be modified.
    - Resource plan: Resource plan where the kernel is running. An absolute resource plan path can be specified. Default default: sample-project/inference.
    - GPU: GPU allocation policy. Options include: no, shared, or exclusive. If set to shared, this indicates GPU packing is enabled.
    - Image name: Specifies where kernel images are pulled from. Specified for kernel image patch, or for a custom image.
    - Resources : CPU memory resource settings for a kernel pod. Default: 
       ncpus=0.5,ncpus_limit=2,mem=1024,mem_limit=4096
  - step4, Update the model with the latest JSON file, dlim model updateprofile model_name -f modelA_profile.json

#### Delete an inference service
Ensure that you have downloaded and configured the dlim utility tool
To delete an inference service using the following step,
  - step1, To delete a model, it must be in stopped status. For the model that you want to delete, stop the inference service: dlim model stop model-name -f
  - step2, Run the following to delete the model including AL the model files in the PV: dlim model undelpoy model-name -f

#### Running inference on a deployed model
To running an inference service using the following step,
  - step1, To submit an inference operation on a deployed model, obtain the REST URI for that model. dlim model view model_name
  - step2, Use the curl command to submit the inference job, curl -k -X POST -d 'JSON-input-data' -H "Authorization: Bearer `dlim config -s`" model-REST-URI

#### View deployed models in IBM Watson® Machine Learning Accelerator
View a list of model from the command line interface. 
  - To view all models: dlim model list
  - To view details for a specific model: dlim model view model-name
  - To view the status information for a specific model:dlim model view model-name -s
  - To view the detailed status information for a specific model: dlim model view model-name -s -a

### Hyperparameter optimization and tuning
Hyperparameters are the parameters used to control the deep learning model training process in IBM Watson Machine Learning Accelerator, such as learning rate and optimization function parameters.

Hyperparameter tuning in IBM Watson Machine Learning Accelerator provides a mechanism to search hyperparameters for a model in a user-defined search space automatically.

Search algorithms are the engine to propose hyperparameter combinations used by a model for training. View the list of available hyperparameter search algorithms, see https://www.ibm.com/docs/en/SSFHA8_5.0/us/dli_hpo_default_search_algorithms.html. To add additional search algorithms to IBM Watson Machine Learning Accelerator, they can be added as search algorithm plugins.

You can develop your own search algorithm plugin(https://www.ibm.com/docs/en/SSFHA8_5.0/us/dli_hpo_develop_plugin.html) which you can add to WML Accelerator using the cluster management console, or using the RESTful API. To add a search algorithm to WML Accelerator using the APIs, see https://www.ibm.com/docs/en/SSFHA8_5.0/us/dli_hpo_add_plugin.html.

In order to use hyperparameter tuning using the cluster management console, the models must be configured accordingly, see: https://www.ibm.com/docs/en/SSFHA8_5.0/us/dli_hpo_model_updates_ui.html.

#### Hyperparameter tuning states
  - SUBMITTED: The tuning job is started.
  - RUNNING: The tuning job is running.
  - STOPPED: The tuning job is stopped in the following cases: 
    - The tuning job was stopped by a user while valid experiments were being run 
    - The dlpd service is restarted while the tuning job was not done.
  - FINISHED: The tuning job exits successfully.
  - FAILED: The tuning job encountered an error, either:
All the experiment training failed
The experiment training completed but no valid metrics were returned

#### Experiment training states
  - NOTRUN: The deep learning experiment job was created.
  - SUBMITTED: The deep learning experiment job is submitted to run.
  - FAILED: The experiment training failed if:
    - The deep learning experiment job submission fails.
    - The deep learning experiment job execution fails.
    - The deep learning experiment job status query fails.
  - RUNNING: The deep learning experiment job is running.
  - STOPPED: The dlpd service was restarted while the deep learning experiment job was not done.
  - KILLED: The experiment training is killed if:
    - The deep learning experiment job was killed.
    - The tuning job was stopped.
  - ERROR: The deep learning experiment job reported error.
  - FINISHED: The experiment training exits successfully. 

#### Hyperparameter search algorithms
Hyperparameter search algorithms supported search algorithms, include: Random, Hyperband, Tree-structured Parzen Estimator (TPE), and ExperimentGridSearch. 
  - Random algorithm proposes hyperparameter combinations from input search space uniformly. It supports one parameter(RandomSeed) to control the search process.
    - RandomSeed: The random seed that is used to sample hyperparameters in uniform distribution. Optional
  - The Hyperband search algorithm verifies more hyperparameter combinations in a fixed resource budget each round. The resource budget can be training epochs, datasets, training time, or other similar resources that affect training process.
    - RandomSeed: The random seed used by Hyperband to propose hyperparameter combinations in the first rung of brackets. Optional.
    - Eta: The reduction factor that controls the proportion of configurations that are discarded in each Hyperband bracket. Default value is 3.
    - ResourceName: The parameter name that is used as a resource in Hyperband, normally training epochs or iterations.
    - esourceValue: The maximum resources that can be used by an experiment training.
  - The TPE search algorithm proposes hyperparameter combinations by splitting the observed experiment training into good and bad ones. The sample hyperparameters are based on the good ones prior and bad prior through Expected Improvement. It supports the following parameters to control the search process
    - RandomSeed: The random seed used for the initial warm up hyperparameter combinations and the random generator of Gaussian Mixture Model. Optional.
    - WarmUp: The number of initial warm-up hyperparameter combinations. Must be larger than 2. Default 20.
    - EICandidate: The number of hyperparameter combinations proposed each round as the candidates for Expected Improvement to propose the final one hyperparameter combination. Must be larger than 1. Default is 24.
    - GoodRatio: The fraction to use as good hyperparameter combinations from previous completed experiment training to build the good Gaussian Mixture Model. Must be larger than 0. Default is 0.25.
    - GoodMax: The max number of good hyperparameter combinations from previous completed experiment training to build the good Gaussian Mixture Model. Must be larger than 1. Default is 25.
  - The TPE search algorithm proposes hyperparameter combinations by splitting the observed experiment training into good and bad ones. The sample hyperparameters are based on the good ones prior and bad prior through Expected Improvement. It supports the following parameters to control the search process:
  - ExperimentGridSearch asks for a list of experiments to train with well-defined hyperparameter combinations. It is not a typical search algorithm but providing a mechanism to submit training with different hyperparameter combination through one call.
