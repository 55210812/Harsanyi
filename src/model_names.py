
# extra_config args are separated by "_"
all_model_names = [
    ### nlp models
    # When loading checkpoints, the correct model checkpoint path will be determined by both the model name and the dataset name.
    "Bert-tiny#pretrain", # pretrained bert can be loaded for finetuning
    "Bert-medium#pretrain",
    "Bert-base#pretrain",
    "Bert-large#pretrain",
    "BERTweet#pretrain", # pretrained for sentiment classification
    "OPT-1.3b#pretrain",
    "llama-7b#pretrain",
    "pythia-70m#pretrain",
    "pythia-160m#pretrain",
    "pythia-410m#pretrain",
    "pythia-1b#pretrain",
    "pythia-1.4b#pretrain",
    "pythia-2.8b#pretrain",
    "pythia-6.9b#pretrain",
    "pythia-12b#pretrain",
    # todo: add finetuned models here


    ### image models
    "vgg11#",
    "vgg16#",
    "alexnet#",
    "resnet20#",
    "resnet56#",
    "resnet18#",
    "resnet50#",

    ### tabular models
    "mlp#nlayer=5_width=128",
    "resmlp#nlayer=5_width=128"
]
