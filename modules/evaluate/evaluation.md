
For evaluation, you have a protein language model (pLM) pre-trained on a set of data, and want to test how well its representations capture (transfer to) a downstream task (e.g. predicting a label).
HuggingFace models ship (through a shared interface definition) with the base model (the "trunk") and versions for specific tasks e.g. AutoModelForSequenceClassification. A "head" aka "probe" is a lightweight model, like an MLP or Linear layer, that takes representations from the pLM "trunk" and uses information contained in them to predict the desired label.

This evaluation codebase envisions three main use cases, modeling-wise:
1) evaluation as-is. If you have a model checkpoint that has already been trained or adapted to the evaluation task, you should be able to just run it through the evaluation pipeline.
Requirements:
- Pretrained model (trunk + probe)
- Test set (protein sequences + labels)
- Tokenizer.
- Metrics of interest
Process:
Take your test set protein sequences, tokenize them, run them through the model and get predictions. Then, score those predictions against your test set's labels. Congratulate yourself on your model's performance.

2) probe training. (we anticipate this will be the major use case) If you have a pretrained model and want to test its representations in a range of downstream tasks, you need to train the probes to use the representations to predict the labels.
Requirements:
- Pretrained model (trunk)
- Probe (defined by the user )
- A set of probe-relevant hyperparameters to search through, and an idea of how you want to search (e.g. random, bohb, latin hypercube)
- Training set (protein sequences + labels)
- Validation set (protein sequences + labels)
- Test set (protein sequences + labels)
- Tokenizer
- Metrics of interest
Process:
Take your train, val, and test set protein sequences and run them through the model (trunk) and get embeddings. Ideally, cache these embeddings so that you're not having to generate them each time you search a different combination of hyperparameters. Do a hyperparameter search, each time training your probe on the training set, and scoring its performance on the validation set. Keep track of the hyperparameter settings that give you the best performance on the validation set. Once your search budget is exhausted, train a probe with the best combination of hyperparameters. Take your test set embeddings and run them through the probe to get predictions. Then, score those predictions against your test set's labels. Congratulate yourself on your model's performance.

3) full fine-tuning. If you want to fully adapt your pretrained model to the task at hand, potentially losing some representation generalizability but hopefully making it better for the task at hand.
Requirements:
- Pretrained model (trunk + head)
- A set of relevant hyperparameters to search through, that can apply to the whole model (e.g. learning rate)
- Training set (protein sequences + labels)
- Validation set (protein sequences + labels)
- Test set (protein sequences + labels)
- Tokenizer
- Metrics of interest
Process:
Take your train, val, and test set protein sequences and tokenize them. Do a hyperparameter search, each time training a copy of your model on the training set, and scoring its performance on the validation set. Keep track of the hyperparameter settings that give you the best performance on the validation set. Once your search budget is exhausted, train the model with the best combination of hyperparameters. Take your tokenized test set sequences and run them through the model to get predictions. Then, score those predictions against your test set's labels. Congratulate yourself on your model's performance.
