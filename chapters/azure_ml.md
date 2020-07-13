# Machine learning on Azure

[go back to Home Page :house:](https://github.com/jayavardhankaspa/Azure-ML-Training/blob/master/README.md#contents)

## Contents

* [Intro to Azure Machine Learning](#intro-to-azure-machine-learning)

## Intro to Azure Machine Learning

Azure machine learning has plethora of tools ranging from data store management to Automated ML.

* **Workspace** is the home in the Azure machine learning tools which contains all the tools and quick start tutorials related to Azure Machine Learning

* **Development tools**

  * **Notebooks** sample notebooks and user uploaded developments notebooks can be accessed here
  * **Automated ML** iterates on multiple algorithms and hyper parameters for a choosen set of metrics to find a best model for your requirement
  * **Designer** it is a drap and drop tool that let's the user create machine learning models without actual code. It has the flexibility to either start from stracth or to strat from a pre-existing template for a quick start.
* **Asset management tools**
  * **Datasets** we can create datasets from local files, webfiles, datastores or Azure open datasets
  * **Experiments** helps in organising and streamline the runs. Each run in Azure should be either a new run or should be an associated to any of the previous runs.
  * **Model** models produced from training in the azure and models imported into Azure after training outside Azure environment.
  * **Endpoints** realtime endpoints/API endpoints are instantations of developed models into either a webservice that can be hosted in the cloud or an IoT module for integrated device deployments. They are used for scoring and pipelines for automation. we can select an endpoint and view its details including the scoring URI thorugh the endpoints consile.
    * [Documentation on consuming Azure models using endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service)
* **Resource management tools**
  * **compute** is used to manage computing resources where we run our trainings or host our models. we can manage compute instances, training clusters , inference clusters  and attached compute.
  * **Datastore** are attached dataaccounts which can be used for training or obtaining datasets. It includes accounts of:
    * Azure blob storage
    * Azure file storage
    * Data lake storage Gen1 and Gen2
    * Azure SQL and mySQL databases
  
