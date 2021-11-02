#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import qlib
import ruamel.yaml as yaml
from qlib.utils import init_instance_by_config
import pandas as pd
import torch


# In[2]:


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--seed", type=int, default=1000, help="random seed")
parser.add_argument("--config_file", type=str, default="/home/qlib_test/examples/benchmarks/TRA/us_latest.yaml", help="config file")
parser.add_argument("-f")
args = parser.parse_args()


seed = args.seed
config_file = args.config_file

# set random seed
with open(config_file) as f:
    config = yaml.safe_load(f)

# seed_suffix = "/seed1000" if "init" in config_file else f"/seed{seed}"
seed_suffix = ""
config["task"]["model"]["kwargs"].update(
    {"seed": seed, "logdir": config["task"]["model"]["kwargs"]["logdir"] + seed_suffix}
)

# initialize workflow
qlib.init(
    provider_uri=config["qlib_init"]["provider_uri"],
    region=config["qlib_init"]["region"],
)


# In[3]:


latest_dataset = init_instance_by_config(config["task"]["dataset"])


# In[4]:


model = init_instance_by_config(config["task"]["model"])


# In[5]:


state_dict = torch.load("/home/qlib_test/examples/benchmarks/TRA/output/us_158_nassp_epoch100/model.bin", map_location="cpu")["model"]


# In[6]:


model.model.load_state_dict(state_dict)


# In[7]:


model.fitted = True


# In[8]:


pred = model.predict(latest_dataset)


# In[9]:


# latest_dataset.to_pickle(path="/home/us_dataset.pkl")


# In[10]:


# pred


# In[11]:


reset_df = pred.reset_index('instrument')
reset_df


# In[13]:


# from IPython.display import clear_output
import numpy as np
indexs = reset_df.index.drop_duplicates()
cnt = 0 

for idx in indexs[-1:]:
    # cnt+=1
    # if cnt < 800: continue
    print(idx)
    if type(reset_df.loc[idx]['score']) is np.float32:
        stack = reset_df.loc[idx]
    else:
        stack = reset_df.loc[idx].sort_values(by='score' , ascending=False)
    stack.to_csv('/home/results_2021_11/'+str(idx)+'.csv', sep=',', na_rep='NaN')
    # input()
    # clear_output()


# In[14]:


# reset_df.to_pickle(path="/home/reset_df.pkl")


# In[ ]:




