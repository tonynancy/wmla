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

### Start training with Watson Machine Learning Accelerator

### Inference
Using the elastic distributed inference capability in WML Accelerator, users can create inference services and deploy published models.

#### Download and configure the elastic distributed inference (dlim) command line utility 
  - step1, To use the dlim command, you must to download and obtain the dlim command line utility for Watson Machine Learning Accelerator from IBM Git at https://github.com/IBM/wmla-assets/
  - step2, Add dlim to PATH, for example:PATH=$PATH:/usr/local/bin/dlim
  - step3, Configure dlim: dlim config -c https://cpd_route
  - step4, To run dlim commands, use a prefetched JWT token directly, for example: dlim config -t -u username -k apikey
  - step5, To see what subcommands are available with the dlim command, run the dlim --help command: dlim --help
  - Examples, PATH=$PATH:/usr/local/bin/dlim, dlim config -c https://cpd-cpd-instance.apps.ibm.com, dlim config -t -u username -k apikey, dlim model list
  - Note,If you are running dlim in a Watson Studio notebook, dlim cannot be configured.  To run dlim commands, use a prefetched JWT token directly, For example: dlim model list --jwt-token $USER_ACCESS_TOKEN --rest-server cpd_route. To use a prefetched token, create a local user that contains your Cloud Pak for Data username and API, for example: dlim config -t -u username -k apikey.

#### Create an inference service
Before you begin, make sure that you have the model files that you are creating the inference service for. To create the inference service, you will need to create a kernel file, a model.json file, a readme file, and the model to be deployed.
  - step1, Create a working directory and copy your model files into that directory.
  - step2, Create the model.json file for the model and add it to the directory. 
  - step3, Create a README.md file. A readme file describes what the input data looks like and what type of inference server response is expected. Add this file to your working directory.
  - step4, Create a kernel file, see Create a kernel file for an inference service. The kernel file must be added to your working directory.
  - step5, Deploy the model: dlim model deploy -p working_directory.
  - Note: Elastic distributed inference creates a model service deployment, and the cpd-scheduler allocates memory to run the service deployment. There must be enough GPUs to run the model service for the kernel pods to start. 
  
#### Start an inference service
Before starting the service, you must update the kernel profile in order for the model information to appear in the Profile tab of your deployment. Use the dlim command to update your kernel profile: dlim model updateprofile model-name -f profile.json.
  - step1, Ensure that you have downloaded and configured the dlim utility tool, see https://www.ibm.com/docs/en/SSFHA8_5.0/us/edi-cli.html.
  - step2, Start inferenc service, dlim model start model-name.
  - step3, Verify that the model was started: dlim model view model-name -s.
