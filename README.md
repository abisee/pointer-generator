This repository contains code for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*. The test set output of the models described in the paper can be found [here](https://drive.google.com/file/d/0B7pQmm-OfDv7MEtMVU5sOHc5LTg/view?usp=sharing).

## About this code
This code is based on the [TextSum code](https://github.com/tensorflow/models/tree/master/textsum) from Google Brain.

This code was developed for Tensorflow 0.12, but has been updated to run with Tensorflow 1.0.
In particular, the code in attention_decoder.py is based on [tf.contrib.legacy_seq2seq_attention_decoder](https://www.tensorflow.org/api_docs/python/tf/contrib/legacy_seq2seq/attention_decoder), which is now outdated.
Tensorflow 1.0's [new seq2seq library](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention) probably provides a way to do this (as well as beam search) more elegantly and efficiently in the future.

## How to run

### Get the dataset
To obtain the CNN / Daily Mail dataset, follow the instructions [here](https://github.com/abisee/cnn-dailymail). Once finished, you should have [chunked](https://github.com/abisee/cnn-dailymail/issues/3) datafiles `train_000.bin`, ..., `train_287.bin`, `val_000.bin`, ..., `val_013.bin`, `test_000.bin`, ..., `test_011.bin` (each contains 1000 examples) and a vocabulary file `vocab`.

**Note**: If you did this before 7th May 2017, follow the instructions [here](https://github.com/abisee/cnn-dailymail/issues/2) to correct a bug in the process.

### Run training
To train your model, run:

```
python run_summarization.py --mode=train --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

This will create a subdirectory of your specified `log_root` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**Warning**: Using default settings as in the above command, both initializing the model and running training iterations will probably be quite slow. To make things faster, try setting the following flags (especially `max_enc_steps` and `max_dec_steps`) to something smaller than the defaults specified in `run_summarization.py`: `hidden_dim`, `emb_dim`, `batch_size`, `max_enc_steps`, `max_dec_steps`, `vocab_size`. 

**Increasing sequence length during training**: Note that to obtain the results described in the paper, we increase the values of `max_enc_steps` and `max_dec_steps` in stages throughout training (mostly so we can perform quicker iterations during early stages of training). If you wish to do the same, start with small values of `max_enc_steps` and `max_dec_steps`, then interrupt and restart the job with larger values when you want to increase them.

#### Run training with flip
By default training is done with "teacher forcing",
instead of generating a new word and then feeding in that word as input when
generating the next word, the expected word in the actual headline is fed in.

However, during decoding the previously generated word is fed in when
generating the next word. That leads to a disconnect between training
and testing. To overcome this disconnect, during training
you can set a random fraction of the steps to be replaced with
the predicted word of the previous step. You can do this with `--flip=<prac>`

You can increase `flip` in a scheduled way. First train without any flip and then
increade flip to `0.2`
(https://arxiv.org/abs/1506.03099)

For debugging, if you want to see what are all the predicted words for all steps, run with `--mode=flip`

### Run (concurrent) eval
You may want to run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:

```
python run_summarization.py --mode=eval --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job.

The eval job will also save a snapshot of the model that scored the lowest loss on the validation data so far.

### Run beam search decoding
To run beam search decoding:

```
python run_summarization.py --mode=decode --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

Note: you want to run the above command using the same settings you entered for your training job (plus any decode mode specific flags like `beam_size`).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.

Additionally, the decode job produces a file called `attn_vis_data.json`. This file provides the data necessary for an in-browser visualization tool that allows you to view the attention distributions projected onto the text. To use the visualizer, follow the instructions [here](https://github.com/abisee/attn_vis).

If you want to run evaluation on the entire validation or test set and get ROUGE scores, set the flag `single_pass=1`. This will go through the entire dataset in order, writing the generated summaries to file, and then run evaluation using [pyrouge](https://pypi.python.org/pypi/pyrouge). (Note this will *not* produce the `attn_vis_data.json` files for the attention visualizer).

By default the beamsearch algorithm takes the best `--topk` results but instead you can
specificy that the `topk` result are randomly selected using `--temperature` parameter. (e.g. `0.8`)

You can request multiple results to be generated for each article with `--ntrials`.
You can force the different trials to be different using `--dbs_lambda` (e.g. `11`)
to add
a penality for having a beam with same token as another beam. (https://arxiv.org/pdf/1610.02424.pdf)

### Evaluate with ROUGE
`decode.py` uses the Python package [`pyrouge`](https://pypi.python.org/pypi/pyrouge) to run ROUGE evaluation. `pyrouge` provides an easier-to-use interface for the official Perl ROUGE package, which you must install for `pyrouge` to work. Here are some useful instructions on how to do this:
* [How to setup Perl ROUGE](http://kavita-ganesan.com/rouge-howto)
* [More details about plugins for Perl ROUGE](http://www.summarizerman.com/post/42675198985/figuring-out-rouge)

**Note:** As of 18th May 2017 the [website](http://berouge.com/) for the official Perl package appears to be down. Unfortunately you need to download a directory called `ROUGE-1.5.5` from there. As an alternative, it seems that you can get that directory from [here](https://github.com/andersjo/pyrouge) (however, the version of `pyrouge` in that repo appears to be outdated, so best to install `pyrouge` from the [official source](https://pypi.python.org/pypi/pyrouge)).

### Tensorboard
Run Tensorboard from the experiment directory (in the example above, `myexperiment`). You should be able to see data from the train and eval runs. If you select "embeddings", you should also see your word embeddings visualized.
