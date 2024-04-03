# Trung Vo's Homework Session 7

## Sambanova
Changed --ntask from 16 to 8
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
