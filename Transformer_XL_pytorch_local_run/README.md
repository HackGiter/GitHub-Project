# Run Transformer-XL locally and adapt to the Newest pytorch version

##### the Original Project : https://github.com/kimiyoung/transformer-xl

##### the Description of this files:

all of these files are copied from the original project. But the original one is a project which is designed for multi-gpu platform and the datasets like WikiText-103 are too big for laptop or personal desktop which did not have a powerful gpu like GTX1060. Intending to run the project on my laptop(with MX350 and Intel 7U 8gen), I change the parameters of the project.And I run with a newest version of pytorch on linux, so I change some places in files. And I add some personal notations for myself. Hope this can help you too.

#### How to run

Please open the terminal in the folder where local_run.sh is in. And type:

##### ./local_run.sh train

It will run the code and begin train

##### ./local_run.sh eval

It will begin to evaluate the model

### For more detail, please refer the original project or the orginal essay:https://arxiv.org/abs/1901.02860