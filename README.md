# Kubernetes Scheduler submission guide
This document is a guide designed to help users launch their containerized deep learning jobs with the new Kubernetes-based scheduler. The scheduler's purpose is to exploit the iterative nature of a deep learning training task in order to perform per epoch profiling and optimize the allocation of GPU memory to the running jobs. The scheduler optimizes resources use, allocating an arbitrary number of GPUs to a job, in order to efficiently use all the GPUs on the machine. With the help of snapshots, taken at the end of each epoch, it is also possible to preempt running jobs in order to execute higher priority jobs. Since your script has to be aware of being run with the scheduler, a python library has been developed to encapsulate the interaction with the scheduler's components.

**Disclaimer**: Please use the scheduler only to submit experiments and not developing/debug scripts. The scheduler does not support python notebooks, since it assumes a relatively constant GPU memory usage. For non-DL jobs or jobs that do not use GPU memory, please make sure your script interacts with the scheduler as described in this guide.

## Submission with run-scheduler

If you have your docker image on westworld (with the right modifications, see below), then you can start your job by running the command run-scheduler:
`run-scheduler [--image_name IMAGE_NAME] [--script_path SCRIPT_PATH] [--custom_dir CUSTOM_DIR] [--snapshots_dir SNAPSHOTS_DIR] [--batch_size BATCH_SIZE] [--training_set_size TRAINING_SET_SIZE]
                     [--max_epochs MAX_EPOCHS] [--is_single_gpu IS_SINGLE_GPU] [--disable_gpu_sharing DISABLE_GPU_SHARING] [--estimated_memory ESTIMATED_MEMORY] [--port_mapping PORT_MAPPING] [--est_nhours EST_NHOURS]
                     job_name `

Arguments semantic:

 - “**job_name**”: the name of your experiment

- “**image_name**”: the name of your docker image.

- “**script_path**”: the string representing the path of the script inside the docker image, that will be run using the command “python *script_path*”. Please note that this path is a container path so if you plan to run a script into a the custom folder mounted with the "*custom_dir*" parameter, you have to take into account the location where it will be mounted. For example if your script is in "**/home/myname/tests/test1.py**" and you mount into the container as *custom_dir* "**/home/myname/tests**" please note that your script path is **not** "**/home/myname/tests/test1.py**" anymore, the *custom_dir* is mounted at **/custom** so you have to specify "**/custom/test1.py**" as *script_path*.

- “**custom_dir**”: path to a directory to which you have access that you want to mount inside the running container at /custom. You can also run your script from here.

- “**snapshots_dir**”: path to a directory to which you have access that will be mounted inside the running container at /snapshots. The library saves here snapshots of your training in case it's stopped .

- “**batch_size**”: batch size used in the training for the experiment. It will be used by the scheduler along with the training_set_size to compute the number of iterations in one epoch, it will be exported as environment variable inside the running container as **BATCH_SIZE**

- “**training_set_size**”: training set size for the experiment, excluding the validation set. It is used by the scheduler along with the batch size to compute the number of iterations in one training epoch. It is exported as enviroment variable inside the running container as **TRAINING_SET_SIZE**

- “**max_epochs**”: represents the maximum number of epochs for your job. It is used by the scheduler to predict the total duration of your job. It is exported as **EPOCHS_TO_BE_DONE**

- “**is_single_gpu**”: tells the scheduler that your code doesn’t support multi-GPU training. If it’s true, the scheduler never schedules your job on more than 1 GPU.

- “**disable_gpu_sharing**”: tells the scheduler that your script cannot be run in sharing on a GPU. If set to true, the job will always be scheduled alone on a GPU, it’ll be more difficult to find the space to schedule it so it may remain pending for longer time. It is advised to set it false, since from preliminary experiments it has been shown the total throughput with GPU sharing is higher, set it to true only if you really need some constraints about interference.

- “**estimated_memory**”: the estimated GPU memory used by your **running container [in MiB].** Note that this is usually larger than the model size, and it’s suggested to overestimate it because if it’s too small the job may fail. 

- “**port_mapping**”: a string \<nodePort1\>:\<containerPort1\>,\<nodePort2\>:\<containerPort2\>,... to expose container ports. 

- “**est_nhours**”: estimation of the duration of your job (in hours), must be >1. It will be adjusted at the end of the first epoch with profiling values 

Additionally, your container will automatically mount your dataset folders (/home/\<username\>/datasets) in /data


In case of errors, the logs will be available by using the command `scheduler-get-logs [jobID]` that will return the output of your script, that is saved in a text file by the scheduler.

**Note**: CPU allocation is different from reservations made on the Google Sheet file. The scheduler handles a shared pool of cores, your script can spawn multiple processes (e.g. pytorch num_workers) and they will be allocated on this shared pool.

## Other commands

- `scheduler-state [--all_users] [--nlast] [--extended]` will return the descriptions and the status of the `nlast` jobs submitted. Use `--all_users` to switch between your jobs or all jobs and  `--extended` to display all options

- `scheduler-cancel-job [jobID]` is used to forcefully end a submitted job

- `scheduler-get-logs [jobID]` is used to show the logs for a particular job. If it's pending for a long time it's suggested to kill it.

# Requirements
As anticipated, your docker image and your script need to be slightly modified in order to be run with the scheduler efficiently

## Docker image build
In your Dockerfile you should add the installation of the dedicated API library for the scheduler:

 ```RUN pip install -e "git+https://github.com/mknnj/kube-apilibrary/#egg=kubernetesdlprofile" ```

<!---Then you should build the image and tag it:

 ```docker tag myimage:latest kube-master-node:31320/myimage:version ```

Then you must push it to the private docker registry in the cluster, in order to have it available on all the cluster’s nodes:

 ```docker push kube-master-node:31320/myimage:version ```

This is because the scheduler may schedule your job on nodes different from the node on which you built you image.-->

## Script modification

If you want to run a Python deep learning script with the scheduler, you must include the dedicated API library in your code.

In every script you run in your image you must include the API library:

`from kubernetesdlprofile import kubeprofiler`

This python library is used to encapsulate the necessary interactions between your script and the scheduler. For example the scheduler may preempt your job: The script has to know where to save the snapshot, then wait for the epoch to finish and then when it's executed again. It has to know that it's being restarted, where to find the snapshot to restart with. Another example is that when the script ends, it signals the scheduler that has been completed without problems. 

This library helps you with this by hiding this complexity behind simple function calls to send signals and profile data to the scheduler. Its instance will have attributes that will contain values specified by you in the cli command.

You have to use it in your code by instantiating an object and calling its methods in some points of your code.
[ you can find working examples for Pytorch,  Pytorch Lightning and Tensorflow at the end of this guide ]
The methods the API object exposes are:
* **measure()** that has to be called at the end of each training step
*  **start_epoch()** that has to be called at the start of each epoch
*  **end_epoch()** that has to be called at the end of each epoch (including validation)

These methods are used in order to profile and collect data about the duration of training steps and epochs. In this way, the scheduler's optimizer knows about the performance of your job in the different GPU configurations. 
In addition, GPU memory occupation of your job is measured in order to allow GPU sharing without running out of memory.

The scheduler exposes in your container some environment variables, in order to handle distributed training and parametrize some values. Some of the environment variables exposed by the scheduler are saved by the API instance object in some of its attributes.
Here you have the complete list of environment variables that are exposed, and the name of the attribute of the API instance object in which the value is saved:
|Name  | Usage | Name of API attribute
|--|--|--|
| MASTER_PORT | Used by PyTorch distributed techniques | \ |
| WORLD_SIZE | Used by PyTorch distributed techniques | \ |
| MASTER_ADDR | Used by PyTorch distributed techniques | \ |
| NODE_RANK | Used by PyTorch distributed techniques | RANK |
| TF_CONFIG| Used by TensorFlow distributed techniques | \ |
| JOB_UUID | Unique id of the job | JOBID |
| JOB_RESTARTED | Tells if the job needs to be restarted from a checkpoint (converted to bool) | RESTARTED |
| RESERVED_MEM | Memory reserved for this job [Mib] | \ |
| EPOCHS_DONE | Epochs already completed by the job | EPOCHS_DONE  |
| HEARTBEAT_FREQ | Number of iteration measured before sending profile data in a measure() call | PROFILE_EACH |
| NODELIST | List of nodes on which the job has been scheduled | NODE_LIST |
| NUM_GPUS| Number of GPUs assigned to the job | NUM_GPUS |
| NUM_NODES| Number of nodes assigned to the job (same as WORLD_SIZE) | NUM_NODES |
| PL_TORCH_DISTRIBUTED_BACKEND| Set to "nccl" by default | \ |


**Variables set by the user (same value as specified in the CLI command, use them to parametrize your job):** 


|Name  | Usage | Name of API attribute
|--|--|--|
| BATCH_SIZE | Batch size as set by the user | BATCH_SIZE|
| EPOCHS_TO_BE_DONE | Max epochs as set by the user | EPOCHS |
| TRAINING_SET_SIZE| Number of training samples in the dataset as set by the user | TRAINING_SET_SIZE |

The value of these variables is in the api instance and is simple to extract and use in your code:

    api = kubeprofiler.KubeProfiler()
You can access a saved environment variables as:

    api.SAVEDVARIABLENAME
For example if you want to get the batch size you specified for your job in the run-scheduler command you can get it in your code as:

    api.BATCH_SIZE
---
Many of these variables are not needed in your code, but some of them have to be handled properly in order to have your script work nicely with the scheduler. 
In your code it's recommended to use
* BATCH_SIZE
* EPOCHS
* TRAINING_SET_SIZE

Those are values you have specified in the CLI command and you can get them in your code from the API Instance. It is recommended to use their values in your code because:
* You already specified them in the run-scheduler command, you can easily control them from there if you need to.
* Their value is used by the optimizer to estimate the remaining time for your job. If you don't use the values from the environment variables you need to be careful and make sure you use the same values in your code or otherwise these estimation won't be truthful. Your job may be heavily penalized by being preempted or given less resources if it exceedes the estimated completion time.

---
### Job restarting
The scheduler may stop and restart your job, in order to restore the correct running state of your job you need checkpoint and checkpoint loading. It's recommended to exploit the API Instance attributes:
* RESTARTED: bool to check that tells if you need to reload a checkpoint
* EPOCHS_DONE: number of epochs already done, used to restore training state in TensorFlow
* root_dir and ckpt_path: are relative paths generated using the jobid, used to save checkpoints. 

You can write your custom paths for saving and reloading checkpoint but be sure that if your training is suddenly stopped by the scheduler, it can be resumed by the checkpoint you specify.

---
Environment variables used in distributed training are not required to be handled by your code, but they are necessary since Tensoflow and Pytorch use them behind the scenes. If you want your job to work on more than 1 GPU by the scheduler remember to set *is_single_gpu* parameter to *false* and then add to your script a line of code to support multi-GPU (do not worry about NUM_NODES too much for now):


For Example:
in Pytorch Lightning you have to specify the number of gpus and the number of nodes (take the value from NUM_NODES and NUM_GPUS):
```python

# nn.prof is the API instance specified in the network definition
trainer = pl.Trainer(accelerator="gpu", devices = nn.prof.NUM_GPUS, num_nodes=nn.prof.NUM_NODES, strategy="ddp", max_epochs=nn.profiler.EPOCHS, profiler="simple", default_root_dir=nn.prof.root_dir)

```
.. in Tensorflow:
```python

# this will get the distributed configuration from TF_CONFIG (set by the scheduler)
# remember to use with  mirrored_strategy.scope(): for the model definition
mirrored_strategy = tf.distribute.MirroredStrategy()

```

**Note**: NUM_NODES is for multi-node training but since we are testing on just one machine, you do not have to worry about it too much.

If your job doesn't specify any distributed strategy support please set *isSingleGPU* to *true* as command parameter, in this way your job won't be scheduled on more than 1 GPU.
## Complete Examples
Here you have some toy examples in each framework, adding the scheduler approach to some hello-world scripts. These are simple working example to show you the usage of the library and where to call its key functions in order to properly interact with the scheduler.
### Pytorch Lightning
 ```dockerfile
 FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN pip install pytorch-lightning==1.5.10 torchvision

RUN pip install -e "git+https://github.com/mknnj/kube-apilibrary/#egg=kubernetesdlprofile"

COPY works /home/works

WORKDIR /

CMD ["python","/home/works/profilerexample.py"]
 ```

     works/profilerexample.py:

 ```python
 import torch
from torch import nn
import torch.nn.functional as  F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.models as  models
import pytorch_lightning as  pl
from kubernetesdlprofile import kubeprofiler
class TestNN(pl.LightningModule):
   def  __init__(self):
      super().__init__()
      self.model = models.mobilenet_v2()
      self.prof = kubeprofiler.KubeProfiler()
def forward(self, x):
   out = self.model(x)
   return  out
def training_step(self, batch, batch_idx):
   x, y = batch
   x_hat = self.model(x)
   loss = F.nll_loss(x_hat, y)
   self.log("train_loss", loss)
   self.prof.measure()
   return  loss
def on_train_epoch_start(self):
   self.prof.start_epoch()
def on_train_epoch_end(self):
   self.prof.end_epoch()
def  configure_optimizers(self):
   optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
   return  optimizer
if  __name__=="__main__":
   dataset = CIFAR10("/data/cifar", download=True, transform=transforms.ToTensor())
   nn = TestNN()
   train, val = random_split(dataset, [nn.prof.TRAINING_SET_SIZE, len(dataset)-nn.prof.TRAINING_SET_SIZE])
   batch_size = nn.prof.BATCH_SIZE
   epochs = nn.prof.EPOCHS
   workers = 10
   trainer = pl.Trainer(accelerator="gpu", devices = nn.prof.NUM_GPUS, num_nodes=nn.prof.NUM_NODES, strategy="ddp", max_epochs=epochs, profiler="simple", default_root_dir=nn.prof.root_dir)
   if  not nn.prof.RESTARTED:
      trainer.fit(nn, DataLoader(train, batch_size=batch_size, num_workers= workers, persistent_workers=True, drop_last=True))
   else:
      trainer.fit(nn, DataLoader(train, batch_size=batch_size, num_workers= workers, persistent_workers=True, drop_last=True), ckpt_path=nn.prof.ckpt_path)
  ```

### Tensorflow
 ```dockerfile
 FROM nvcr.io/nvidia/tensorflow:21.09-tf2-py3

RUN pip install -e "git+https://github.com/mknnj/kube-apilibrary/#egg=kubernetesdlprofile"

COPY works /home/works

WORKDIR /

CMD ["python","/home/works/profilerexample.py"]
 ```

     works/profilerexample.py:

 ```python
 import tensorflow as tf
import os
from kubernetesdlprofile import kubeprofiler
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#uses TF_CONFIG to initialize distributed
mirrored_strategy = tf.distribute.MirroredStrategy()
with  mirrored_strategy.scope():
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10)
   ])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
prof = kubeprofiler.KubeProfiler()
checkpoint_filepath = os.path.join(prof.root_dir, "tf_ckpts")
print("chekpoint dir: ", checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
   filepath=checkpoint_filepath,
   save_weights_only=True,
   monitor='accuracy',
   mode='max',
   save_best_only=True)
class KubeProfilerCallback(tf.keras.callbacks.Callback):
   def __init__(self, prof):
      super(KubeProfilerCallback, self).__init__()
      self.prof = prof
   def on_epoch_begin(self, epoch, logs=None):
      self.prof.start_epoch()
   def on_epoch_end(self, epoch, logs=None):
      self.prof.end_epoch()
   def on_train_batch_end(self, batch, logs=None):
      self.prof.measure()
kube_callback = KubeProfilerCallback(prof)
model.compile(optimizer='adam',
   loss=loss_fn,
   metrics=['accuracy'])
batch_size = prof.BATCH_SIZE
epochs = prof.EPOCHS
RESTARTED = prof.RESTARTED
initial_epoch = 0
if RESTARTED:
   model.load_weights(checkpoint_filepath)
   initial_epoch = prof.EPOCHS_DONE
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, callbacks=[model_checkpoint_callback, kube_callback])
model.evaluate(x_test, y_test, verbose=2)
  ```

## Tensorboard
To set up tensorboard make your script save tensorboard logs in the custom folder then start a container with tensorboard to expose some port:

 ```docker run -v </custom_dir/path/to/saved/tf/logs>:/data -d -p 31346:6006 <tf image> tensorboard --logdir /data ```
 
 A tf image could be nvcr.io/nvidia/tensorflow:21.09-tf2-py3.

Remember to kill the container when the job is complete.

### PyTorch
 ```dockerfile
 FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN pip install torchvision

RUN pip install -e "git+https://github.com/mknnj/kube-apilibrary/#egg=kubernetesdlprofile"

COPY works /home/works

WORKDIR /

CMD ["python","/home/works/profilerexample.py"]
 ```

     works/profilerexample.py:

 ```python
import torch
import torch.nn as  nn
import torch.optim as  optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from kubernetesdlprofile import kubeprofiler
class  RandomDataset(Dataset):
   def __init__(self, size, length):
      self.len = length
      self.data = torch.randn(length, size)
   def __getitem__(self, index):
      return  self.data[index]
   def __len__(self):
      return self.len
class Model(nn.Module):
   def __init__(self, input_size, output_size):
      self.prof = kubeprofiler.KubeProfiler()
      super(Model, self).__init__()
      self.fc = nn.Linear(input_size, output_size)
      self.sigmoid = nn.Sigmoid()
   def forward(self, input):
      return self.sigmoid(self.fc(input))
if __name__ == '__main__':
   input_size = 5
   output_size = 1
   model = Model(input_size, output_size)
   batch_size = model.prof.BATCH_SIZE
   data_size = model.prof.TRAINING_SET_SIZE
   PATH = model.prof.root_dir+'/cifar_net.pth'
   rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
   batch_size=batch_size, shuffle=True)
   print("Let's use", torch.cuda.device_count(), "GPUs!")
   model_par = nn.DataParallel(model).cuda()
   print(f"Scheduler assigned {model.prof.NUM_GPUS} gpus")
   optimizer = optim.SGD(params=model_par.parameters(), lr=1e-3)
   cls_criterion = nn.BCELoss()
   if model.prof.RESTARTED:
      model_par.load_state_dict(torch.load(PATH))
   for epoch in range(model.prof.EPOCHS - model.prof.EPOCHS_DONE):
      model.prof.start_epoch()
      for data in rand_loader:
         targets = torch.empty(data.size(0)).random_(2).view(-1, 1)
         if torch.cuda.is_available():
            input = Variable(data.cuda())
            with torch.no_grad():
               targets = Variable(targets.cuda())
         else:
            input = Variable(data)
            with torch.no_grad():
               targets = Variable(targets)
         output = model_par(input)
         optimizer.zero_grad()
         loss = cls_criterion(output, targets)
         loss.backward()
         optimizer.step()
         model.prof.measure()
      model.prof.end_epoch()
      torch.save(model_par.state_dict(), PATH)
```
