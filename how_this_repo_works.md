This document records how this repo trains model and the specifics of each components such as model arch, configs, optimizer, checkpointing.

# configuration
`tools/cfgs/docs.yaml` contains all possible configurations and explanations of each. Listed here are some preset common cfgs to be appended to the command to override the yaml files.

```shell script
# for training with adam and stepwise lr decay
MODEL.TRAIN.OPTIMIZATION.OPTIMIZER adam \
MODEL.TRAIN.OPTIMIZATION.LR 0.0001 \
MODEL.TRAIN.OPTIMIZATION.DECAY_STEP_LIST '[20, 40, 60]' \
```

# Optmizer
- SGD
    - can be used with weight decay or momentum
- Adam
    - recall adam can be thought of as sgd with moment + RMSprop
- Adam One Cylce
    - haha I am not sure how it works but it was the default
    
