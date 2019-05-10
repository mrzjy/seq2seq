[TOC]



# Seq2Seq Baseline Implementation (Using Tensorflow Estimator)



## Description
This repo contains Python + Tensorflow implementation of classical seq2seq models (listed as follows)

| Model    | Code Reference             | Content                                                         |
| ----------- | ---------------- | ------------------------------------------------------------ |
| RNN    | https://github.com/tensorflow/nmt           | RNN (+ Attention)                                                |
| Transformer | https://github.com/tensorflow/models/tree/master/official/transformer           | Attention is all you need     |


## Version
- Python 3
- Tensorflow 1.13.0 (should be compatible with older or 2.0 version)


## Hyperparameters
| Name    | Usage | Description             |
| ----------- | ----------- | ---------------- |
| num_bucket    | Data   | Number of data buckets to generate (put data of variable length into buckets) |
| bucket_width | Data | Length interval of bucket |
| max_seq_len | Data | Max sequence length for input data (both source and target sequence) |
| data_dir | Data | Directory where train/test/valid data is located |
|train_x| Data | Data file name of source sequence, default="train_x"; same for test_x, valid_x|
|train_y| Data | Data file name of target sequence, default="train_y"; same for test_y, valid_y|
|vocab_file| Data | Data file of vocab, located in data_dir by default, (**MAKE SURE you read FAQ data format**) |
|model_dir| Data | Directory where model is saved or loaded |
| model | Model | Model type, 3 choices are: trans (Transformer), rnn, and other strings are rnn+attention|
|log_frequency|Train|Control the logging info frequency (e.g., 200 means print log every 200 steps(batches))|
|reverse_sequence|Train|Whether or not reverse your target sequence (hence learn the generation reversly)|
More model-specific or training-specific parameters are left for you to explore.

## Command line (example)
- In order to **start training**:

~~~Python
python main.py --hparams=data_dir=my_data_dir,model=trans
~~~
- In order to **perform inference**:

~~~Python
python predict.py --hparams=model=rnn,model_dir=saved_model/S2S,pred_x=path/to/inputs,infer_mode=beam_search
~~~

Set get_all_beams=True, then you could get all beam search results printed in the console.

- In order to **export a model** (e.g., the best model saved during training):

~~~Python
python export.py --hparams=model=rnn,model_dir=saved_model/S2S/best
~~~

Then the exported model will show up in the exported_model/rnn by default. (Make sure you input the right model type: model=rnn while model_dir=path/to/Transformer will raise error)

- In order to conveniently **check out the exported model** input/output, use saved_model_cli provided by Tensorflow

~~~Python
saved_model_cli show --dir $exported_model_dir --tag_set serve --signature_def serving_default
~~~

(e.g., exported_model_dir=exported_model/rnn/1556183667)

Then, the console should return similar information as below:

~~~
The given SavedModel SignatureDef contains the following input(s):
  inputs['source'] tensor_info:
      dtype: DT_STRING
      shape: (-1, -1)
      name: source:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['inputs'] tensor_info:
      dtype: DT_INT32
      shape: (-1, -1)
      name: Cast:0
  outputs['inputs_len'] tensor_info:
      dtype: DT_INT32
      shape: (-1)
      name: sub:0
  outputs['predicts'] tensor_info:
      dtype: DT_INT64
      shape: (-1, -1)
      name: ReverseSequence:0
  outputs['predicts_len'] tensor_info:
      dtype: DT_INT32
      shape: (-1)
      name: S2SA/Decoder/decoder/while/Exit_13:0
Method name is: tensorflow/serving/predict
~~~
- In order to **serve the model**

There are various ways to acheive this. I personally recommand the tensorflow/serving way. One of the best feature I found is its support of server-side batching (see the official tutorial for yourself), which is indeed helpful for server performance. Or if you want to implement this feature by yourself, refer to the working BERT-as-service example by Hanxiao (https://github.com/hanxiao/bert-as-service).


## FAQ
- **Language?**

The code is originally used for training on Chinese language, hence there are no tools for subwords for example. Feel free to customize your preprocessing in utils/Data_utils.py

- **What data format?**

By default, all input data is considered present in one data_dir, as follows:

my_data_dir

├── train_x

├── train_y

├── test_x

├── test_y

├── vocab

For x or y files, each line is a **tokenized (space-splitted)** sequence (e.g. 你 今天 好 吗 ？)

However, if you set char=True, then each line will be tokenized into chars, no matter it's already tokenized ot not:

(e.g, "你 今天 好 吗" will be processed to "你 今 天 好 吗")

For vocab file, each line is a token, **make sure you set the first 3 lines** as:

&lt;PAD&gt;

&lt;EOS&gt;

&lt;UNK&gt;

since PAD_id, EOS_id and UNK_id are by default 0, 1 and 2.

- **Why Estimator API?**

I found it quite handy after I get used to this TF api, not only for training management, but also for later exportation or tensorflow serving for example. Basically, all we need are just input_fn and model_fn, and when you want to explore something in the middle of training, you could always use a training hook for your customization.

- **Why TF data iterator?**

Because there are many useful functions such as bucketing, buffered reading, parallelism, support for distributed training, etc. Of course one could always use their own data helper. But currently TF data iterator meets all the need that I have, especially when combined with python generator, which means that I could do data preprocessing as I want in a Python style before sending the data to TF data iterator pipeline.

- **Found some tricks?**

Yes, there are a few tricks used in the code, as also used in the official transformer code, such as:

**Reverse generation (Optional)** : Learn the generation backward might result in better performance

**Weight Tying** : Sharing the weight matrix between input-to-embedding layer and output-to-softmax layertied-weights for softmax

**label smoothing regularization**: Penalizing low entropy output distribution brings better performance