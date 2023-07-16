# What is Bolden?

This model was created to honor Marie C. Bolden, who was a spelling bee champ:

* https://en.wikipedia.org/wiki/1908_National_Education_Association_Spelling_Bee
* https://www.bbc.com/news/world-us-canada-65755792

# What is the purpose of the Bolden LLM?

It's an example of how to train a model from Scratch, and with a little effort, I may build it into something that is
usable for more than education.

### Example output

Clearly the model is wildly inaccurate, but the output is otherwise fine for a small LLM.
![](example_output.png)

## Basics

### Step 1: Install torch

#### For NVIDIA: Install details for torch + cuda:

Instructions: https://pytorch.org/get-started/locally/

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
`

#### FOR M1 install details for torch + mps:

Instructions at: https://developer.apple.com/metal/pytorch/

`pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
`

#### FOR CPU-only install details for torch

This probably won't end well, but here's how you do the base install of pytorch:

`pip3 install --pre torch torchvision torchaudio
`

### Step 2: Install other requirements

`pip3 install -r requirements.txt`

### Step 3: Update the scripts

Update the following variables at the top of train_bolden:

* device_name_: "cuda", "mps", or "cpu" (I've only tested "cuda")
* context_length_: Bigger means using more memory. You probably won't be able to handle a big context.
* dataset_column_: Name of the column in the dataset to train off of.
* dataset_name_: Name of the huggingface training dataset.

### Step 4: Train

Note: make sure you updated the variables at the top of the script!

`python3 train_bolden.py`

If it doesn't fit in memory, you may need to reduce the size of the hidden layers or the number of hidden layers.

### Step 5: See the results

Note: make sure you updated the variables at the top of the script!

`python3 run_bolden.py`