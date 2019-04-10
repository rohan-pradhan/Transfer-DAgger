NAMING CONVENTION 

args.train_layers == 1 means we are freezing the evaluation layers
args.train_layers == 2 means we are freezing the convolution layers
rc refers to road color

A directory train_few_shot_TRANSFER_1_rc_3 would mean training with few shot transfer learning having the evaluation layers frozen with a road color of 0.3

A directory train_few_shot_TRANSFER_2_rc_55 would mean training with few shot transfer learning having the convolution layers frozen with a road color of 0.55

A directory train_one_shot_TRANSFER_1_rc_5 would mean training on one shot transfer learning having the evaluation layers frozen with a road color of 0.5

A directory train_one_shot_TRANSFER_2_rc_6 would mean training on one shot transfer learning having the convolution layers frozen with a road color of 0.6

A directory train_full_6 would mean training an entire network without freezing any layers with a color of 0.6

A directory train_one_shot_FULL_rc_3 would mean training an entire network without freezing any layers on one shot transfer learning with a road color of 0.3

The directories train and val are the original train and validation directories 
