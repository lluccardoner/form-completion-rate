# Form completion rate prediction

This project consist of:
* A pipeline to train and deploy a model that predicts forms expected completion rates (defined as submissions over views)
* A simple http API to serve online predictions with the deployed model

## Usage

Run API server:
```bash
```

Train and deploy model:

```bash
```

Send requests to server:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @resources/sample.json
```

## Project information

This project uses Apacke Spark ML python [package](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html).
It has been chosen because I am familiar with it and because it allows to save and load a full trained [ML pipeline](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html). 
The benefit of that is that the real-time service that serves the predictions is agnostic of all the data transformations needed.

### Dataset

The dataset given has the following columns:
* form_id: the id of the form
* views: the number of views of the form
* submissions: the number of submissions fo the form
* feature columns: 47 unnamed feature columns "feat_XX"

The target to be predicted is the completion rate, defined as submissions over views.

The assumed use case is to predict the form completion rate before the form is released. 
Therefore, it is assumed that only the feature columns should be used as features to train the model.
The view and submissions columns will only be used to compute the target value.

All features have numerical values and it will be assumed that all features are numeric.
Checking the features distribution, it can be seen that they are very skewed and in some cases with some outliers.

See notebooks/data_exploration.ipynb for the dataset analysis.

### Processing

The processing pipeline consists of:
* A vector assembler that groups all used features in one vector

### Modeling

As a model, a Linear Regression model is used. It is easy to interpret, simple and fast to train.

To choose the best model, a grid search is done to find the best hyper parameters.
3-fold cross validation is used to evaluate the model and choose the best one.

### Evaluation

As mentioned in the previous section, the model is evaluated using a K-fold cross validation during training.
In the src/app.py the model is also evaluated on a testing set. 
Then the metrics are stored in the output folder along with a plot of the residuals.

### Deployment

The src/deployment.py trains the model using grid-search and cross-validation. 
Then the full pipeline is saved to be used by the real-time service. 

### Monitoring

Monitoring is not implemented in this project but it is also an essential part of the ML pipeline.

Once the model is deployed there are few things that could be monitored:
* Predictions of the model
* Predict time of the model
* Size of the model
* Parameters of the model
* History of models used

Regarding the real-time service, the infrastructure metrics that could be monitored are:
* Uptime: the % of time the service is available
* CPU and memory usage of the service
* Requests per minute
* The average and maximum latency
* Errors of the service

Other product related metrics such as the API usage growth would be also useful.

### Running the service on the cloud

Amazon Web Services is an option of running the service in the cloud. 
It has many ML services to manage end-to-end ML pipes such as SageMaker.

Model training, optimization, tuning and evaluation can be done in batch time running in a virtual machine (EC2). 
Another option is to user an EMR cluster to leverage the Spark distributed computation in case the dataset is bigger.

For model deployment we could use S3 with versioning or CodeDeploy to keep track of the models trained.

The real-time service can run in a single instance. 
If needed, the real-time service could be replicated in multiple regions i.e US, EU, ASIA to serve requests faster.
A load balancer can be added in front to manage requests and to be able to scale.

AWS also has the CloudWatch service that provides monitoring and logging.

Other services that could be used include CloudFormation to manage the infrastructure 
or Amazon VPC to control the networking environment between instances.

## Future work

* Grid Search using value distributions instead of discrete values
* Add log-transform to the processing pipeline
* Train model with full dataset
* Dockerize both services
* Do API stress test
* Finish tests