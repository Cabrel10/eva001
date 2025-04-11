 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 60, 5)]      0           []                               
 cnn_module (Functional)        (None, 60, 64)       13376       ['input_1[0][0]']                
 global_average_pooling1d       (None, 64)           0           ['cnn_module[0][0]']             
 ga_module (Functional)         (None, 60, 64)       384         ['input_1[0][0]']                
 repeat_vector (RepeatVector)   (None, 60, 64)       0           ['global_average_pooling1d[0][0]']
 lstm_module (Functional)       (None, 60, 64)       17920       ['input_1[0][0]']                
 concatenate (Concatenate)      (None, 60, 192)      0           ['ga_module[0][0]',              
                                                                  'repeat_vector[0][0]',          
                                                                  'lstm_module[0][0]']            
 dense_1 (Dense)                (None, 60, 3)        579         ['concatenate[0][0]']            
==================================================================================================
Total params: 32,259
Trainable params: 32,259
Non-trainable params: 0