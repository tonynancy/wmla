# Training deep models on IBM Cloud Pak for Data

IBM Cloud Pak for Data is a cloud-native solution that enables you to put your data to work quickly and efficiently.

Your enterprise has lots of data. You need to use your data to generate meaningful insights that can help you avoid problems and reach your goals.

But your data is useless if you can't trust it or access it. Cloud Pak for Data lets you do both by enabling you to connect to your data, govern it, find it, and use it for analysis. Cloud Pak for Data also enables all of your data users to collaborate from a single, unified interface that supports many services that are designed to work together.

To train a deep learnin model, you can either run the training code in a Jupyter notebook or deep learning experiment builder. But deep learning exeriment builder provides more advanced training capability, such as using multiple GPUs across mulitple nodes to run distributed training, and training models with elastic nodes.

## Training deep learning models using the deep learning experiment builder

To build deep learning experiments, you must have access to Watson Machine Learning Accelerator, which is not installed by default as part of IBM Watson Machine Learning. An administrator must install Watson Machine Learning Accelerator on the IBM Cloud Pak for Data platform and provide you with user access.

As a data scientist, you need to train thousands of models to identify the right combination of data in conjunction with hyperparameters to optimize the performance of your neural networks. You want to perform more experiments and faster. You want to train deeper neural networks and explore more complicated hyperparameter spaces. IBM Watson Machine Learning accelerates this iterative cycle by simplifying the process to train models in parallel with an auto-allocated GPU compute containers.

### Steps to create deep learning experiment

In side a project on ibm cloud pak for data, choose to create a new `deep learning` asset with below steps.

    1. Set the name for the experiment and an optional description.

    2. Specify the location of your training data using one of these options:

        - Storage volume: Use if training data is available on a mounted storage volume.

        - Training data files using a relative folder path: Use if training data is uploaded to a local repository. The system adminstrator will have to set up a Watson Machine Learning Accelerator PVC.

        - Select data in the project: Use if reading or downloading training data from a data asset or connection. If using a data assset or data connection, see below supported data connections for training data.

    3. If choose to select data in the project, for data source in file types, you can choose to pre-download it to several locations:

        - Training worker local storage

        - Training worker shared storage

        - Storage volume, if you also selected any storage volume as training data location.

    4. Associate a model definition. For details, refer to below associating model definitions.

    5. Click Create and run to create the model definition and run the experiment.

### Supported data connections for training data

Watson Machine Learning Accelerator supports the following data connection that you can use as the training when creating deep learning experiment.

1. IBM services

For more details on adding connected data assets, including IBM services, see Adding data from a connection.

- IBM Cloud® Databases for MySQL

- IBM® Cloud Object Storage

- IBM Cloudant®

- IBM Cloud Data Engine

- IBM Cognos® Analytics

- IBM Cloud Databases for MongoDB

- IBM Db2® for z/OS®

- IBM Db2 Warehouse

- IBM Netezza® Performance Server

2. Third-party services

For more details on adding connected data assets, including third-party services, see Adding data from a connection.

- Amazon RDS for MySQL

- Amazon RDS for Oracle

- Amazon S3

- Apache Cassandra

- Apache Hive

- Microsoft Azure File Storage

- Microsoft Azure Data Lake Storage

- Microsoft SQL Server

- MongoDB

- MySQL

- PostgreSQL

- SingleStoreDB

- Snowflake

- Teradata

### Associating model definitions

You must associate one or more model definitions to the deep learning experiment. You can associate multiple model definitions as part of running an experiment. Model definitions can be a mix of either existing model definitions or ones that you create as part of the process.

- Click Add model definition.

- Choose whether to create a new model definition or use an existing model definition.

   - To create a new model definition, click Create model definition and specify the options for the model definition. For details, refer to Creating a new model definition.

   - To choose an existing model definition, and set experiment attributes. For details, see: Select an existing model definition.

- Optionally, choose to use a global execution command which will override model definition values. If selected, in the Global execution command box, enter the execution command that can be used to execute the Python code. The execution command must reference the Python code and can optionally pass the name of training files if that is how your experiment is structured.

### Creating a new model definition

1. Type a unique name and a description.

2. Upload a model training .zip file that has the Python code that you have set up to indicate the metrics to use for training runs. 

3. Set the execution command. This command runs when the model is created.

4. Specify attributes to be used during the experiment.

    - From the Framework box, select the appropriate framework. This must be compatible with the code you use in the Python file.

    - From the Resrouce type box, either CPU or GPU.

    - From the Hardware specification box, choose the number and size of CPUs and GPUs allocated for running the experiment with one of these options.

        - Select from the list of predefined software specifications.

        - Create a custom hardware specification. 

            - For CPU training, set the number of CPU units and memory size. 

            - For GPU training, set the number of GPU resources per worker. Set the GPU resource type to either: Slice, Generic, Full. For Slice type, you can then select one availabe MIG (Multi-Instance GPU) profile.

    - From the Training type box, select the training type.

        - If CPUs are selected, single node and multi node is available.

        - If GPUs are selected, single node, multi node, and elastic nodes are available.

### Selecting an existing model definition

1. Select an existing model definition and select the attributes used during the experiment.

2. From the Framework box, select the appropriate framework. This must be compatible with the code you use in the Python file.

3. From the Resrouce type box, either CPU or GPU.

4. From the Hardware specification box, choose the number and size of CPUs and GPUs allocated for running the experiment with one of these options.

    - Select from the list of predefined software specifications.

    - Create a custom hardware specification. 

        - For CPU training, set the number of CPU units and memory size. 

        - For GPU training, set the number of GPU resources per worker. Set the GPU resource type to either: Slice, Generic, Full. For Slice type, you can also select one availabe MIG (Multi-Instance GPU) profile.

5. From the Training type box, select the training type.

    - If CPUs are selected, single node and multi node is available.

    - If GPUs are selected, single node, multi node, and elastic nodes are available.

6. Click Select to create the model definition and run the experiment.

### Monitoring training in deep learning experiment

Open the created deep learning experiment, in the Training Runs tab, select one training and open it. You will see the training progress in real time and when the training run is complete, you can check the logs for each training worker.

### Deploy the trained model

You can use your trained model to classify new images only after the model has been deployed.

To deploy the model, you must first promote it to a deployment space.

1. In the deep learning experiments list, click on your training run.

2. Once the training run is completed, the model appears in the Completed table. Select Save model from the action menu. Give the model a name and click Save. This stores the model in the Watson Machine Learning repository and saves it to the project. In the project's asset you can now see the saved model.

3. To deploy the model, you must first promote it to a deployment space. Choose Promote to space from the action menu for the model. Select Target space or Create a new deployment space, and click Promote.

4. A prompt is shown to navigate to your Deployment Space after promoting the model.

5. After navigating to the deployment space click on the model name.

6. Click New deployment.

7. Choose Online and enter a name for the deployment.

8. Click Create. Monitor the status of the model deployment.

When the deployment is started, you can quickly verify the deployed model with sample input.

## Training deep learning models using Watson Machine Learning Accelerator

In addition to training deep learning models using the deep learning experiment builder, users can also directly training deep learning models using Watson Machine Learning Accelerator.

### Introduction to Watson Machine Learning Accelerator

Watson Machine Learning Accelerator is a deep learning service for data scientists on Cloud Pak for Data.

Watson Machine Learning Accelerator can be connected to Watson Machine Learning to take advantage of the multi-tenant resource plans that manage resource sharing across Watson Machine Learning projects. With this integration, data scientists can use the Watson Machine Learning Experiment Builder and Watson Machine Learning Accelerator hyperparameter optimization.

Watson Machine Learning Accelerator provides the following benefits on Cloud Pak for Data:

- Distributed deep learning architecture and hyper-parameter search and optimization that simplifies the process of training deep learning models across a cluster for faster time to results.

- Advanced Kubernetes scheduling including consumer resource plans, ability to run parallel jobs and dynamically allocate GPU resources.

- Included ready-to-use deep learning frameworks, such as TensorFlow and PyTorch.

### Supported deep learning frameworks

Supported deep learning frameworks included with IBM Watson® Machine Learning Accelerator.

The following frameworks are included with Watson Machine Learning Accelerator for use with IBM® Watson Machine Learning.

- TensorFlow 2.14.1

- PyTorch 2.1.2

- Tensorboard 2.12.2

- torchVision 0.15.2

- OpenCV 4.7.0

- scikit-learn 1.1.3

- XGBoost 1.7.6

- ONNX 1.13.1

- PyArrow 11.0.0

- Python 3.10.12

Note:

- All frameworks support the supported versions of Python.

- PyTorch supports single node training, distributed training, and elastic distributed training.

- TensorFlow supports single node training, distributed training, and elastic distributed training (with TF.Keras).

### Using Watson Machine Learning Accelerator batch training

- Using data connection through REST API
POST request for REST API /platform/rest/deeplearning/v1/execs accepts a data parameter in string with format specification.

- Using data connection through the command line interface (CLI)
Use the Watson Machine Learning Accelerator CLI. You can download the CLI from the Watson Machine Learning Accelerator console.

When running training, you can specify your connection data source using the --cs-datastore-meta and --data-source options. If both options are specified, --data-source will be used.

### Application types and states in Watson Machine Learning Accelerator

Use this information to learn about available application types and application states in IBM Watson® Machine Learning Accelerator.

1. Application types

The following application types are available in Watson Machine Learning Accelerator.

- Training: Applications that run training workloads.

- Notebook: Applications that run notebook workloads.

- Inference: Applications that run inference workloads. These are long running deployed models for online production use.

2. Application states

The following application states are available in Watson Machine Learning Accelerator. 

- Submitted: A job is submitted successfully and the job specifications are being reviewed.

- Pending: Job specifications cannot be met. The job cannot move on to launching state until all specifications are met.

- Launching: A job was started successfully with all specifications met.

- Running: A job is running training based on the job specifications provided.

- Finished: Job training was completed with the specified resources and specifications.

- Killed: Job training was stopped due to a user stopping the application.

- Failed: A job failed during run time due to a job specification that was unmet or issues with resource allocation.

- Error: A job cannot be submitted due to existing errors in the application configuration.

### Using the Watson Machine Learning Accelerator command line interface (CLI) dlicmd tool

Download the Watson Machine Learning Accelerator CLI tool from the Watson Machine Learning Accelerator console. Navigate to Help > Command Line Tools to download the dlicmd tool.


## Elastic distributed training

Watson Machine Learning Accelerator utilizes elastic distributed training as part of its general distribution deep learning framework. 

The elastic distributed training engine uses a fine grained control for training. This is an alternate to the coarse grained strategy, where the resource utilization is capped and no additional GPUs can be added to the training phase. With elastic distributed training, the scheduler distributes the task and the session scheduler dynamically issues resources from the resource manager, allowing for additional GPUs to be added.

In addition, the following functions are supported with the elastic distributed training engine:

  - Single-node, multi-GPU training support without parameter server .
    
  - Multi-node, multi-GPU training support with parameter server.
    
  - Both synchronous and asynchronous distributed training algorithms support for multi-node, including synchronous gradient data control algorithm, and asynchronous gradient data control algorithm.

  - NVIDIA Collective Communications Library (NCCL) support for broadcasting, reducing gradient and weight data across multiple GPUs.

### Start Elastic distributed training 

To run elastic distributed training, update your training model files to make the following two changes: 

  1, Create a FabricModel instance to define an elastic distributed training model, see FabricModel definition for details on defining a FabricModel.
  
  2, To start training a model, run the train function of FabricModel, see FabricModel methods for details.

### FabricModel definition

To utilize elastic distributed training, update your model to include the FabricModel definition. 

  - model: Required. The model instance, either an instance of tf.keras.model (TensorFlow) or torch.nn.Module (PyTorch)
    
  - datasets_function: Required. A python function which will return train and validation dataset.
    
  - loss_function: Required. Loss function. For TensorFlow, it can be a tf.keras.losses instance or a string of the loss function name. For PyTorch, it must be a callable loss function.
    
  - optimizer: Required. Optimizer for model training. For TensorFlow, it can be a tf.keras.optimizers instance or a string of the loss function name. For PyTorch, it must be a torch.optim instance.
    
  - metrics: List of metrics to be evaluated by the model during training and testing. For TensorFlow, metrics can be a string (name of a built-in function), function or a tf.keras.metrics metric instance. For PyTorch, metrics can be a function or a callable instance.

### FabricModel methods

To start training a model, run the train function of FabricModel:

  - epoch_number: Required. Number of epochs to train the model. Must be an integer.

  - batch_size: Required. Local batch size to use per GPU during training. Must be an integer.

  - engines_number: Optional. Maximum number of GPUs to use during training. Must be an integer.

  - num_dataloader_threads: Optional. Number of threads to load data batches for model training. Must be an integer.

  - validation_freq: Optional. Frequency between how many epochs to run model validation. Default is 1. Must be an integer.

  - checkpoint_freq: Optional. Frequency between how many epochs to save model checkpoint. Default 1. Must be an integer.

  - effective_batch_size: Optional. Global batch size across all workers and it is exclusive with engines_number. Must be an integer.

 
### Running elastic distributed training

There is not much difference to launch elastic distributed training compared with other regular training on Watson Machine Learning Accelerator, except it requires some particular values for training interfaces.

  1, Using data connection through REST API
  - POST request to REST API /platform/rest/deeplearning/v1/execs to start training workload.    

  2, Using data connection through the command line interface (CLI). You can download CLI from https://github.com/IBM/wmla-assets/releases/download/v4.8.2/dlicmd.py. The dlicmd.py can execute deep learning tasks using cluster resources, assumes that models can access data sources from within the cluster, model data must either be dynamically downloaded, reside on shared directories, or be available from remote data connection services. 
  - Use the Watson Machine Learning Accelerator CLI(dlicmd.py).

    
## Hyperparameter optimization

Watson Machine Learning Accelerator features hyperparameter tuning and optimization.

Hyperparameters are parameters whose values are set before starting the model training process. Deep learning models, including convolutional neural network (CNN) and recurrent neural network (RNN) models can have anywhere from a few hyperparameters to a few hundred hyperparameters. The values specified for these hyperparameters can impact the model learning rate and other regulations during the training process as well as final model performance.

Watson Machine Learning Accelerator uses hyperparameter optimization algorithms to automatically optimize models. The algorithms used include Random Search, Tree-structured Parzen Estimator (TPE) and Bayesian optimization based on the Gaussian process. These algorithms are combined with a distributed training engine for quick parallel searching of the optimal hyperparameter values. 

### Hyperparameter optimization supported search algorithms

Supported search algorithms, include: Random, Hyperband, Tree-structured Parzen Estimator (TPE), and ExperimentGridSearch. 

  - Random algorithm proposes hyperparameter combinations from input search space uniformly. 

  - The Hyperband search algorithm verifies more hyperparameter combinations in a fixed resource budget each round. The resource budget can be training epochs, datasets, training time, or other similar resources that affect training process.

  - The TPE search algorithm proposes hyperparameter combinations by splitting the observed experiment training into good and bad ones. The sample hyperparameters are based on the good ones prior and bad prior through Expected Improvement. 

  - ExperimentGridSearch asks for a list of experiments to train with well-defined hyperparameter combinations. It is not a typical search algorithm but providing a mechanism to submit training with different hyperparameter combination through one call.


### Hyperparameter optimization tuning states

Tuning states for hyperparameter tuning:

  - SUBMITTED: The tuning job is started.

  - RUNNING: The tuning job is running.

  - STOPPED: The tuning job is stopped in the following cases:

    - The tuning job was stopped by a user while valid experiments were being run

    - The dlpd service is restarted while the tuning job was not done.

  - FINISHED: The tuning job exits successfully.

  - FAILED: The tuning job encountered an error, either: All the experiment training failed The experiment training completed but no valid metrics were returned

### Hyperparameter optimization experiment training states

Experiment training states for hyperparameter:

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

Inference is key features of WML Accelerator, users can create inference services and deploy published models. Inference use dlim as command line tool. Inference use dlim to create, start, stop, delete, edit and view inference service.


## Introduction to inference of WMLA

WML Accelerator has the elastic distributed inference(EDI) capability, users can create inference services and deploy published models. Elastic distributed inference is a secure, robust, scalable inference service which exposes WML Accelerator REST API for users to publish and manage inference services, for REST clients to consume the service, and for administrators to manage the service. An inference service can be used for inference by any authorized client. The elastic distributed inference feature can host models for all projects (or for each line of business). Models are developed and published by developers for a specific project. Published models can run as an inference service. Inference services can be started and stopped.

Inference is key features of WML Accelerator, users can create inference services and deploy published models. Inference use dlim as command line tool. Inference use dlim to create, start, stop, delete, edit and view inference service.

### （EDI）Step to download and configure the elastic distributed inference (dlim) command line utility 

To download and configure an inference dlim using the following step,

  - step1, To use the dlim command, you must to download and obtain the dlim command line utility for Watson Machine Learning Accelerator from IBM Git at https://github.com/IBM/wmla-assets/

  - step2, Add dlim to PATH, for example:PATH=$PATH:/usr/local/bin/dlim

  - step3, Configure dlim: dlim config -c https://cpd_route. where cpd_route is the URL of the Cloud Pak for Data instance that you want to interact with through the command-line interface.

  - step4, To run dlim commands, use a prefetched JWT token directly, for example: dlim config -t -u username -k apikey

  - step5, To see what subcommands are available with the dlim command, run the dlim --help command: dlim --help

  - Examples, PATH=$PATH:/usr/local/bin/dlim, dlim config -c https://cpd-cpd-instance.apps.ibm.com, dlim config -t -u username -k apikey, dlim model list

  - Note,If you are running dlim in a Watson Studio notebook, dlim cannot be configured.  To run dlim commands, use a prefetched JWT token directly, For example: dlim model list --jwt-token $USER_ACCESS_TOKEN --rest-server cpd_route. To use a prefetched token, create a local user that contains your Cloud Pak for Data username and API, for example: dlim config -t -u username -k apikey.

### （EDI）Step to create an inference service

Before you begin, make sure that you have the model files that you are creating the inference service for. To create the inference service, you will need to create a kernel file, a model.json file, a readme file, and the model to be deployed.

To create an inference service using the following step,

  - step1, Create a working directory and copy your model files into that directory.

  - step2, Create the model.json file for the model and add it to the directory. 

  - step3, Create a README.md file. A readme file describes what the input data looks like and what type of inference server response is expected. Add this file to your working directory.

  - step4, Create a kernel file, see Create a kernel file for an inference service. The kernel file must be added to your working directory.

  - step5, Deploy the model: dlim model deploy -p working_directory.

  - Note: Elastic distributed inference creates a model service deployment, and the cpd-scheduler allocates memory to run the service deployment. There must be enough GPUs to run the model service for the kernel pods to start. 

  

### （EDI）Step to start an inference service

Before starting the service, you must update the kernel profile in order for the model information to appear in the Profile tab of your deployment. Use the dlim command to update your kernel profile: dlim model updateprofile model-name -f profile.json.

To start an inference service using the following step,

  - step1, Ensure that you have downloaded and configured the dlim utility tool, see https://www.ibm.com/docs/en/SSFHA8_5.0/us/edi-cli.html.

  - step2, Start inferenc service, dlim model start model-name.

  - step3, Verify that the model was started: dlim model view model-name -s.

### （EDI）Step to stop an inference service

To stop an inference service using the following step,

  - Run the following command:dlim model stop model-name. where model-name is the name of the model that you want to stop. 

### （EDI）Step to edit an inference service

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

### （EDI）Step to delete an inference service

To delete an inference service using the following step,

  - step1, To delete a model, it must be in stopped status. For the model that you want to delete, stop the inference service: dlim model stop model-name -f

  - step2, Run the following to delete the model including ALL the model files in the PV: dlim model undelpoy model-name -f

### （EDI）Step to run inference on a deployed model

To running an inference service using the following step,

  - step1, To submit an inference operation on a deployed model, obtain the REST URI for that model. dlim model view model_name

  - step2, Use the curl command to submit the inference job, curl -k -X POST -d 'JSON-input-data' -H "Authorization: Bearer `dlim config -s`" model-REST-URI

### （EDI）Step to view deployed models in inference service

View a list of model from the command line interface. 

  - To view all models: dlim model list

  - To view details for a specific model: dlim model view model-name

  - To view the status information for a specific model:dlim model view model-name -s

  - To view the detailed status information for a specific model: dlim model view model-name -s -a
