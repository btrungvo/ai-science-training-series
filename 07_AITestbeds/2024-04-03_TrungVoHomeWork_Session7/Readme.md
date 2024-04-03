# Trung Vo's Homework Session 7

## Sambanova
Changed --ntask from 16 to 8, there was an error after iteration 504
```
^MIteration:  20%|█▉        | 500/2549 [07:29<29:04,  1.17it/s]2024-04-03 09:39:54,427 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - epoch:0|local_step:501|global_step:501|average_loss:8.34513|step_loss:8.34513|step_ns_loss:0.65576|step_mlm_loss:7.68937|learning_rate:7.00e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
^MIteration:  20%|█▉        | 501/2549 [07:30<29:03,  1.17it/s]2024-04-03 09:39:55,277 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - epoch:0|local_step:502|global_step:502|average_loss:8.23843|step_loss:8.23843|step_ns_loss:0.59900|step_mlm_loss:7.63943|learning_rate:7.01e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
^MIteration:  20%|█▉        | 502/2549 [07:31<29:01,  1.18it/s]2024-04-03 09:39:56,127 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - epoch:0|local_step:503|global_step:503|average_loss:8.25822|step_loss:8.25822|step_ns_loss:0.61925|step_mlm_loss:7.63898|learning_rate:7.03e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
^MIteration:  20%|█▉        | 503/2549 [07:32<29:00,  1.18it/s]2024-04-03 09:39:56,977 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - epoch:0|local_step:504|global_step:504|average_loss:8.19762|step_loss:8.19762|step_ns_loss:0.61887|step_mlm_loss:7.57875|learning_rate:7.04e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
^MIteration:  20%|█▉        | 504/2549 [07:33<28:59,  1.18it/s]2024-04-03 09:39:57,827 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - epoch:0|local_step:505|global_step:505|average_loss:8.23740|step_loss:8.23740|step_ns_loss:0.61180|step_mlm_loss:7.62560|learning_rate:7.06e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
2024-04-03 09:39:57,835 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - final_loss 8.308523
2024-04-03 09:39:57,844 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1536235 - info     - {'e2e_train_time': 457.0915858745575, 'training_sequences_per_second': 289619.6825559826, 'final_loss': 8.308523178100586, 'training_samples_per_second': 2262.653769968614}
2024-04-03 09:39:57,903 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536238 - info     - NLP app finished
2024-04-03 09:39:57,906 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536242 - info     - NLP app finished
2024-04-03 09:39:57,906 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536236 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
2024-04-03 09:39:57,910 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536241 - info     - NLP app finished
2024-04-03 09:39:57,911 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536240 - info     - NLP app finished
2024-04-03 09:39:57,911 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536237 - info     - NLP app finished
2024-04-03 09:39:57,911 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1536239 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
Traceback (most recent call last):
  File "sambaflow/samba/abexit.py", line 90, in sambaflow.samba.abexit.SambaAtexit.run_with_abexit
  File "/opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py", line 540, in main
    task_module.do_training(args=args,
  File "/opt/sambaflow/apps/nlp/transformers_on_rdu/tasks/lm_tasks/bert_mlperf_lm.py", line 484, in do_training
    mlperf_pretrain.do_training(args, model, optims, model_config,
  File "/opt/sambaflow/apps/nlp/transformers_on_rdu/tasks/lm_tasks/bert_mlperf_trainer.py", line 576, in do_training
    assert training_perf >= args.min_throughput, \
AssertionError: Expected throughput to be at least 560000.0, instead found 289619.6825559826
srun: error: sn30-r1-h1: task 0: Exited with exit code 1
srun: Terminating job step 30399.0
slurmstepd: error: *** STEP 30399.0 ON sn30-r1-h1 CANCELLED AT 2024-04-03T09:40:00 ***
srun: error: sn30-r1-h1: tasks 1,3-5: Terminated
srun: error: sn30-r1-h1: tasks 2,6-7: Killed
srun: Force Terminated job step 30399.0
```
## Graphcore
mnist on Graphcore. Epochs changed from 10 to 30. Accuracy improved from 96.85% to 98.16%
```
srun: job 20544 queued and waiting for resources
srun: job 20544 has been allocated resources
100%|██████████| 9912422/9912422 [00:00<00:00, 315003305.26it/s]
100%|██████████| 28881/28881 [00:00<00:00, 191974158.20it/s]
100%|██████████| 1648877/1648877 [00:00<00:00, 100167886.63it/s]
100%|██████████| 4542/4542 [00:00<00:00, 38408324.13it/s]
Epochs:   0%|          | 0/10 [00:00<?,[08:27:00.611] [poptorch:cpp] [warning] [DISPATCHER] Type coerced from Long to Int for tensor id 10/150 [00:00<?, ?it/s]
                                                       2024-04-03T08:27:01.166470Z PL:POPLIN    208191.208191 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead
Graph compilation:   4%|▍         | 4/100 [00:00<00:05]2024-04-03T08:27:04.609918Z PL:POPLIN    208191.208191 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead
Graph compilation: 100%|██████████| 100/100 [00:21<00:00]2024-04-03T08:27:22.916100Z popart:session 208191.208191 W: Rng state buffer was not serialized.You did not load poplar Engine.Remember that if you would like to run the model using the model runtime then you have to create your own buffer and callback in your model runtime application for rngStateTensor.

Epochs: 100%|██████████| 10/10 [01:51<00:00, 11.14s/it]
  0%|          | 0/125 [00:00<?, ?it/s]                2024-04-03T08:28:52.449755Z PL:POPLIN    208191.208191 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead
                                                       2024-04-03T08:28:54.640388Z PL:POPLIN    208191.208191 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead
Graph compilation: 100%|██████████| 100/100 [00:15<00:00]
 90%|████████▉ | 112/12Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw/train-images-idx3-ubyte.gz
Extracting /home/btrungvo/.torch/datasets/MNIST/raw/train-images-idx3-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting /home/btrungvo/.torch/datasets/MNIST/raw/train-labels-idx1-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting /home/btrungvo/.torch/datasets/MNIST/raw/t10k-images-idx3-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting /home/btrungvo/.torch/datasets/MNIST/raw/t10k-labels-idx1-ubyte.gz to /home/btrungvo/.torch/datasets/MNIST/raw

TrainingModelWithLoss(
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 98.16%
```

## Cerebras
Change batch_size from 1024 to 512
```
2024-04-03 10:50:19,179 INFO:   Effective batch size is 512.
2024-04-03 10:50:19,203 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "model_dir_bert_large_pytorch" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
2024-04-03 10:50:19,205 INFO:   No checkpoints were found in "model_dir_bert_large_pytorch".
2024-04-03 10:50:19,205 INFO:   No checkpoint was provided. Using randomly initialized model parameters.
2024-04-03 10:50:20,478 INFO:   Saving checkpoint at step 0
2024-04-03 10:50:46,729 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_0.mdl
2024-04-03 10:51:01,664 INFO:   Compiling the model. This may take a few minutes.
2024-04-03 10:51:01,666 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-04-03 10:51:02,874 INFO:   Initiating a new image build job against the cluster server.
2024-04-03 10:51:02,963 INFO:   Custom worker image build is disabled from server.
2024-04-03 10:51:02,970 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-04-03 10:51:03,242 INFO:   Initiating a new compile wsjob against the cluster server.
2024-04-03 10:51:03,339 INFO:   compile job id: wsjob-jthhdzshylufgvvnkuy6ra, remote log path: /n1/wsjob/workdir/job-operator/wsjob-jthhdzshylufgvvnkuy6ra
2024-04-03 10:51:13,374 INFO:   Poll ingress status: Waiting for job service readiness.
2024-04-03 10:51:43,385 INFO:   Poll ingress status: Waiting for job ingress readiness.
2024-04-03 10:51:53,397 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-04-03 10:51:57,224 INFO:   Pre-optimization transforms...
2024-04-03 10:52:02,794 INFO:   Optimizing layouts and memory usage...
2024-04-03 10:52:02,832 INFO:   Gradient accumulation enabled
2024-04-03 10:52:02,833 WARNING:   Gradient accumulation will search for an optimal micro batch size based on internal performance models, which can lead to an increased compile time. Specify `micro_batch_size` option in the 'train_input/eval_input' section of your .yaml parameter file to set the gradient accumulation microbatch size, if an optimal microbatch size is known.

2024-04-03 10:52:02,836 INFO:   Gradient accumulation trying sub-batch size 8...
2024-04-03 10:52:07,803 INFO:   Exploring floorplans
2024-04-03 10:52:15,562 INFO:   Exploring data layouts
2024-04-03 10:52:27,285 INFO:   Optimizing memory usage
2024-04-03 10:53:15,180 INFO:   Gradient accumulation trying sub-batch size 64...
2024-04-03 10:53:22,195 INFO:   Exploring floorplans
2024-04-03 10:53:32,144 INFO:   Exploring data layouts
2024-04-03 10:53:52,825 INFO:   Optimizing memory usage
2024-04-03 10:54:26,759 INFO:   Gradient accumulation trying sub-batch size 32...
2024-04-03 10:54:31,938 INFO:   Exploring floorplans
2024-04-03 10:54:38,810 INFO:   Exploring data layouts
2024-04-03 10:54:55,352 INFO:   Optimizing memory usage
2024-04-03 10:55:27,622 INFO:   Gradient accumulation trying sub-batch size 128...
2024-04-03 10:55:34,820 INFO:   Exploring floorplans
2024-04-03 10:55:45,002 INFO:   Exploring data layouts
2024-04-03 10:56:03,325 INFO:   Optimizing memory usage
2024-04-03 10:56:30,645 INFO:   Gradient accumulation trying sub-batch size 256...
2024-04-03 10:56:36,726 INFO:   Exploring floorplans
2024-04-03 10:56:51,929 INFO:   Exploring data layouts
2024-04-03 10:57:18,329 INFO:   Optimizing memory usage
2024-04-03 10:58:00,742 INFO:   Exploring floorplans
2024-04-03 10:58:04,183 INFO:   Exploring data layouts
2024-04-03 10:58:37,848 INFO:   Optimizing memory usage
2024-04-03 10:59:11,806 INFO:   No benefit from gradient accumulation expected. Compile will proceed at original per-box batch size 512 with 6 lanes

2024-04-03 10:59:11,855 INFO:   Post-layout optimizations...
2024-04-03 10:59:24,668 INFO:   Allocating buffers...
2024-04-03 10:59:27,104 INFO:   Code generation...
2024-04-03 10:59:40,722 INFO:   Compiling image...
2024-04-03 10:59:40,727 INFO:   Compiling kernels
2024-04-03 11:03:01,896 INFO:   Compiling final image
2024-04-03 11:05:52,581 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_8939750200954608837
2024-04-03 11:05:52,636 INFO:   Heartbeat thread stopped for wsjob-jthhdzshylufgvvnkuy6ra.
2024-04-03 11:05:52,640 INFO:   Compile was successful!
2024-04-03 11:05:52,648 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2024-04-03 11:05:55,173 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-04-03 11:05:55,452 INFO:   Initiating a new execute wsjob against the cluster server.
2024-04-03 11:05:55,558 INFO:   execute job id: wsjob-o7hidxzx662ek7mtbsqid6, remote log path: /n1/wsjob/workdir/job-operator/wsjob-o7hidxzx662ek7mtbsqid6
2024-04-03 11:06:05,593 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: current job is top of queue but likely blocked by running jobs, 1 execute job(s) running using 1 system(s), 1 compile job(s) running using 67Gi memory. For more information, please run 'csctl get jobs'.
2024-04-03 11:06:15,579 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job is scheduled.
2024-04-03 11:06:25,600 INFO:   Poll ingress status: Waiting for job service readiness.
2024-04-03 11:06:45,638 INFO:   Poll ingress status: Waiting for job ingress readiness.
2024-04-03 11:06:55,660 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-04-03 11:06:55,796 INFO:   Preparing to execute using 1 CSX
2024-04-03 11:07:24,629 INFO:   About to send initial weights
2024-04-03 11:07:56,932 INFO:   Finished sending initial weights
2024-04-03 11:07:56,935 INFO:   Finalizing appliance staging for the run
2024-04-03 11:07:56,955 INFO:   Waiting for device programming to complete
2024-04-03 11:09:58,034 INFO:   Device programming is complete
2024-04-03 11:09:58,929 INFO:   Using network type: ROCE
2024-04-03 11:09:58,930 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2024-04-03 11:09:58,953 INFO:   Input workers have begun streaming input data
2024-04-03 11:10:15,732 INFO:   Appliance staging is complete
2024-04-03 11:10:15,737 INFO:   Beginning appliance run
2024-04-03 11:10:32,958 INFO:   | Train Device=CSX, Step=100, Loss=9.39062, Rate=2985.58 samples/sec, GlobalRate=2985.59 samples/sec
2024-04-03 11:10:50,500 INFO:   | Train Device=CSX, Step=200, Loss=8.70312, Rate=2945.45 samples/sec, GlobalRate=2951.76 samples/sec
2024-04-03 11:11:07,844 INFO:   | Train Device=CSX, Step=300, Loss=7.79688, Rate=2949.38 samples/sec, GlobalRate=2951.84 samples/sec
2024-04-03 11:11:25,382 INFO:   | Train Device=CSX, Step=400, Loss=7.39062, Rate=2931.44 samples/sec, GlobalRate=2943.68 samples/sec
2024-04-03 11:11:42,931 INFO:   | Train Device=CSX, Step=500, Loss=7.80469, Rate=2923.12 samples/sec, GlobalRate=2938.42 samples/sec
2024-04-03 11:12:00,219 INFO:   | Train Device=CSX, Step=600, Loss=7.53125, Rate=2946.17 samples/sec, GlobalRate=2942.25 samples/sec
2024-04-03 11:12:17,610 INFO:   | Train Device=CSX, Step=700, Loss=7.35156, Rate=2944.90 samples/sec, GlobalRate=2942.51 samples/sec
2024-04-03 11:12:35,069 INFO:   | Train Device=CSX, Step=800, Loss=7.27344, Rate=2937.49 samples/sec, GlobalRate=2941.26 samples/sec
2024-04-03 11:12:52,543 INFO:   | Train Device=CSX, Step=900, Loss=7.35938, Rate=2933.05 samples/sec, GlobalRate=2940.02 samples/sec
2024-04-03 11:13:10,091 INFO:   | Train Device=CSX, Step=1000, Loss=7.12500, Rate=2923.82 samples/sec, GlobalRate=2937.77 samples/sec
2024-04-03 11:13:10,092 INFO:   Saving checkpoint at step 1000
2024-04-03 11:13:45,167 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_1000.mdl
2024-04-03 11:14:23,347 INFO:   Heartbeat thread stopped for wsjob-o7hidxzx662ek7mtbsqid6.
2024-04-03 11:14:23,354 INFO:   Training completed successfully!
2024-04-03 11:14:23,354 INFO:   Processed 512000 sample(s) in 174.282130137 seconds.
```
## Groq
Changed hyperparameter led to error so I kept the script the same as original.
```
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████| 346/346 [00:00<00:00, 1.85MB/s]
vocab.txt: 100%|█████████████████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 10.2MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████| 112/112 [00:00<00:00, 1.71MB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████| 760/760 [00:00<00:00, 11.0MB/s]
pytorch_model.bin: 100%|███████████████████████████████████████████████████████████| 17.6M/17.6M [00:00<00:00, 80.6MB/s]
/home/btrungvo/miniconda3/envs/groqflow/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()



Building "bert_tiny"
    ✓ Exporting PyTorch to ONNX
    ✓ Optimizing ONNX file
    ✓ Checking for Op support
    ✓ Converting to FP16
    ✓ Compiling model
    ✓ Assembling model

Woohoo! Saved to ~/.cache/groqflow/bert_tiny
Preprocessing data.
/home/btrungvo/miniconda3/envs/groqflow/lib/python3.10/site-packages/datasets/load.py:1461: FutureWarning: The repository for sst contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/sst
You can avoid this message in future by passing the argument `trust_remote_code=True`.
Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
  warnings.warn(
Downloading builder script: 100%|██████████████████████████████████████████████████| 9.13k/9.13k [00:00<00:00, 38.6MB/s]
Downloading readme: 100%|██████████████████████████████████████████████████████████| 6.68k/6.68k [00:00<00:00, 27.0MB/s]
Downloading data: 100%|████████████████████████████████████████████████████████████| 6.37M/6.37M [00:01<00:00, 5.27MB/s]
Downloading data: 100%|██████████████████████████████████████████████████████████████| 790k/790k [00:00<00:00, 1.33MB/s]
Generating train split: 100%|█████████████████████████████████████████████| 8544/8544 [00:00<00:00, 11813.50 examples/s]
Generating validation split: 100%|█████████████████████████████████████████| 1101/1101 [00:00<00:00, 2021.68 examples/s]
Generating test split: 100%|███████████████████████████████████████████████| 2210/2210 [00:00<00:00, 3903.36 examples/s]

Info: No inputs received for benchmark. Using the inputs provided during model compilation.
Running inference on GroqChip.
Running inference using PyTorch model (CPU).
100%|██████████████████████████████████████████████████████████████████████████████| 2210/2210 [00:04<00:00, 449.47it/s]
+--------+----------+-------------------------+----------------+----------------------+-------------+
| Source | Accuracy | end-to-end latency (ms) | end-to-end IPS | on-chip latency (ms) | on-chip IPS |
+--------+----------+-------------------------+----------------+----------------------+-------------+
|  cpu   |  77.47%  |           2.23          |     449.37     |          --          |      --     |
|  groq  |  77.47%  |           0.05          |    18771.99    |         0.03         |   37576.72  |
+--------+----------+-------------------------+----------------+----------------------+-------------+
Proof point /home/btrungvo/groqflow/proof_points/natural_language_processing/bert/bert_tiny.py finished!
```
