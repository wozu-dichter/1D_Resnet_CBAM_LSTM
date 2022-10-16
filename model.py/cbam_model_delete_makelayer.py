

from multiprocessing import pool
from pandas import concat
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential
import tensorflow as tf


# from attention_module import simAM

class ChannelAttention(layers.Layer):
    def __init__(self, channel, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)

        # self.inputs = inputs
        # channel = self.inputs.shape[-1]
        
        self.channel = channel  #输入数据的最后一维, 即输入通道
        self.avg_pool = layers.GlobalAveragePooling1D()  

        self.max_pool = layers.GlobalMaxPool1D()

        self.share_layer_one = layers.Dense(channel//ratio, 
                                            activation='relu',
                                            kernel_initializer='he_normal',
                                            use_bias=True,
                                            bias_initializer='zeros')

        self.share_layer_two = layers.Dense(channel,
                                            kernel_initializer='he_normal',
                                            use_bias=True,
                                            bias_initializer='zeros')
        
        self.add = layers.Add()
        self.act = layers.Activation('sigmoid')
    
    def call(self, inputs):
        self.inputs = inputs

        avg_pool = self.avg_pool(self.inputs)   # input: [N, time_step, C] output: [N, C]
        avg_pool = layers.Reshape((1, self.channel))(avg_pool) # input: [N, C] output: [N, 1, C]

        max_pool = self.max_pool(self.inputs)   # input: [N, time_step, C] output: [N, C]
        max_pool = layers.Reshape((1, self.channel))(max_pool) # input: [N, C] output: [N, 1, C]

        avg_pool = self.share_layer_one(avg_pool) # input [N, 1, C] output: [N, 1, C/ratio]
        avg_pool = self.share_layer_two(avg_pool) # input [N, 1, C/ratio] output: [N, 1, C]
        max_pool = self.share_layer_one(max_pool) # input [N, 1, C] output: [N, 1, C/ratio]
        max_pool = self.share_layer_two(max_pool) # input [N, 1, C/ratio] output: [N, 1, C]
        
        cbam_feature = self.add([avg_pool, max_pool]) # input: [N, 1, C] ouput: [N, 1, C]
        cbam_feature = self.act(cbam_feature)      
        # print(cbam_feature.shape)

        ## return 广播机制: [N, 1, C]*[N, time_step, C]=[N, time_step, C]
        return layers.multiply([self.inputs, cbam_feature])  ## 等同于*, 广播机制: 两个数组的后缘维度相同, 或者在其中一方的维度为1。广播在缺失或者长度为1的维度上进行补充
        

class SpatialAttention(layers.Layer):
    def __init__(self,  kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

        self.kernel_size = kernel_size # 卷积核的大小
        # self.inputs = inputs

        self.concatenate = layers.Concatenate(axis=2) # 在最后一维, 即通道进行拼接

        self.conv = layers.Conv1D(filters=1,
                                  kernel_size=self.kernel_size,
                                  strides=1,
                                  padding='same',
                                  activation='sigmoid',
                                  kernel_initializer='he_normal',
                                  use_bias=False)
    def call(self, inputs):
        self.inputs = inputs

        avg_pool = keras.backend.mean(self.inputs, axis=2, keepdims=True) # input: [N, time_step, C] ouput: [N, time_step, 1]
        max_pool = keras.backend.max(self.inputs, axis=2, keepdims=True)  # input: [N, time_step, C] ouput: [N, time_step, 1]

        concat = self.concatenate([max_pool, avg_pool]) # ouput: [N, time_step, 2]

        cbam_feature = self.conv(concat)     # input: [N, time_step, 2] ouput: [N, time_step, 1]

        assert cbam_feature.shape[-1] == 1   


        # return 广播机制: [N, time_step, 1] * [N, time_step, C] = [N, time_step, C]
        return layers.multiply([self.inputs, cbam_feature])
        



## basic_residual_block 
class Residual_Bottleneck(layers.Layer):
    def __init__(self, channel, strides=1, two_times=False, downsample=None, **kwargs):
        super(Residual_Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(channel, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv1') 
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv1/BatchNorm')
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv1D(channel, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv2')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.Activation('relu')
       
       # 判断通道数是变化两倍还是四倍
        if two_times == False and channel!=256:
            self.conv3 = layers.Conv1D(channel*4, kernel_size=1, strides=strides, padding='same', use_bias=False, name='conv3')
            # self.simam = simAM.simam_module()
            self.ca = ChannelAttention(channel*4)
            self.sa = SpatialAttention()
            
        elif two_times == True and channel!=256:
            self.conv3 = layers.Conv1D(channel*2, kernel_size=1, strides=strides, padding='same', use_bias=False)
            self.ca = ChannelAttention(channel*2)
            self.sa = SpatialAttention()
        else:
            self.conv3 = layers.Conv1D(channel, kernel_size=1, strides=strides, padding='same', use_bias=False)
            self.ca = ChannelAttention(channel)
            self.sa = SpatialAttention()
            
            # self.simam = simAM.simam_module()
        
        
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv3/BatchNorm')
        self.relu3 = layers.Activation('relu')
        

        self.downsample = downsample
    
    def call(self, inputs):
        residual = inputs
        if self.downsample is not None:
            residual = self.downsample(inputs)

        x = self.conv1(inputs) #input: [N, time_step, C] output: [N, time_stpe, C]
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)       #input: [N, time_step, C] output: [N, time_step ,C]
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)     #input: [N, tiem_step, C] if two_time==True: output: [N, time_step/2, C*2] else: output[N, time_step, C*4] 

        # x = self.lstm(x)
        # x = tf.expand_dims(x, axis=1)             #input:[N, 256] ouput:[N, 1, 256]     
        
        # x = self.simam(x)
        x = self.ca(x)
        x = self.sa(x)

        out = residual + x    #input: [N, time_step, C] output: [N, time_step, C]
        out = self.bn3(out) 
        out = self.relu3(out)
        return out

class spp(layers.Layer):
    def __init__(self):
        super(spp, self).__init__()

        self.maxpool_5 = layers.MaxPool1D(pool_size=5, strides=1, padding='same')
        self.maxpool_9 = layers.MaxPool1D(pool_size=9, strides=1, padding='same')
        self.maxpool_13 = layers.MaxPool1D(pool_size=13, strides=1, padding='same')

    def call(self, x):
        spp_x = x
        x5 = self.maxpool_5(x)
        x9 = self.maxpool_9(x)
        x13 = self.maxpool_13(x)
        y = layers.Concatenate(axis=-1)([spp_x, x5, x9, x13])
        
        return y
    
class sppf(layers.Layer):
    def __init__(self):
        super(sppf, self).__init__()

        self.conv1 = Sequential([
                               layers.Conv1D(128, kernel_size=1, strides=1, padding='same'),
                               layers.BatchNormalization(),
                               layers.Activation('relu')])
        self.maxpool = layers.MaxPool1D(pool_size=5, strides=1, padding='same')

    def call(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)

        x = layers.Concatenate(axis=-1)([x, x1, x2, x3])

        return x




## 1D-resnet layer
def _make_layer(block, channel, strides=1, two_times=False, name=None):
    downsample = None

    # 如果stride!=1, 则最终的残差块的downsample部分(x + residual的residual部分), 计算公式output_timesteps = input_timestep/strides
    # 也就是基本残差块的最终output: H/2, W/2, channel*2, [N, time_steps/2, C*2] 
    # 如果stride=1, 则最终的残差块的downsample部分(x + residual的residual部分), 计算公式output_timmesteps = input_timestep/strides 
    # 也就是基本残差块的最终output: H, W, channel*4, [N, time_steps, C*4] 
    if strides != 1:
        downsample = Sequential([
            layers.Conv1D(channel*2, kernel_size=1, strides=strides, padding='same', use_bias=False, name='conv1'),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name='conv1/BatchNorm') 
        ])
    else:
        downsample = Sequential([
            layers.Conv1D(channel*4, kernel_size=1, strides=strides, padding='same', use_bias=False, name='conv1'),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name='cpnv1/BatchNorm1')
        ])

    layers_list = []

    layers_list.append(block(channel, strides, two_times=two_times, downsample=downsample))
    return Sequential(layers_list, name=name)

def model_simAM():

    input_dim = layers.Input(shape=(81, 1), dtype='float32')  # input:[N, 81, 1]

    x = layers.Conv1D(16, kernel_size=3, strides=1, padding='same', name='conv1')(input_dim) # input: [N, 81, 1]  output:[N, 81, 16]
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv1/BatchNorm')(x)
    x = layers.Activation('relu')(x)

    # 1D-resnet 
    # arg: channel是x的最后一维,即通道数 
    # arg: strides=2, two_times=True: input: [N, time_step, C] ouput:[N, time_step/2, C*2]
    # arg: strides=1, two_times=False: input: [N, time_step, C] ouput:[N, time_step, C*4]
    x = _make_layer(Residual_Bottleneck, channel=16, name='block1')(x)   # input[N, 81, 16] output:[N, 81, 64]
    lstm = x
    lstm = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(lstm) # input: [N, 21, 256] output: [N, 5, 256]
    # print(lstm.shape)
    lstm = layers.LSTM(64, dropout=0.3)(lstm) #input:[N, 5, 256] output:[N, 64] ##具体输出可能有点复杂, 可以直接看layers.LSTM函数
    x1 = tf.expand_dims(lstm, axis=1)       #input:[N, 64] output:[N, 1, 64]


    # x1 = layers.GlobalAveragePooling1D()(x)    #input:[N, 21, 256] output:[N, 256]
    # x1 = tf.expand_dims(x1, axis=1)             #input:[N, 256] ouput:[N, 1, 256]
    # x1 = layers.Concatenate(axis=-1)([lstm, x1]) #input:[N, 1, 64] [N, 1, 256] output:[N, 1, 320]
    
    
    x = _make_layer(Residual_Bottleneck, channel=64, strides=2, two_times=True, name='block2')(x) # input:[N, 81, 64] output:[N, 41, 128]
    lstm = x
    lstm = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(lstm) # input: [N, 21, 256] output: [N, 5, 256]
    # print(lstm.shape)
    lstm = layers.LSTM(64, dropout=0.3)(lstm) #input:[N, 5, 256] output:[N, 64] ##具体输出可能有点复杂, 可以直接看layers.LSTM函数
    x2 = tf.expand_dims(lstm, axis=1)       #input:[N, 64] output:[N, 1, 64]


    # x2 = layers.GlobalAveragePooling1D()(x)    #input:[N, 21, 256] output:[N, 256]
    # x2 = tf.expand_dims(x2, axis=1)             #input:[N, 256] ouput:[N, 1, 256]
    # x2 = layers.Concatenate(axis=-1)([lstm, x2]) #input:[N, 1, 64] [N, 1, 256] output:[N, 1, 320]
    
    x = _make_layer(Residual_Bottleneck, channel=128, strides=2, two_times=True, name='block3')(x) # input:[N, 41, 128] output:[N, 21, 256]
    
    
    lstm = x

    # print(lstm)  ## 如果model_CBAM的模型输入是[N, 779, 30], 那么pool_size=4, 如果是[N, 81, 1], 则pool_size=2, 
    # output: time_step = (time_step-pool_size+1)/strides input: [N, time_step, C]
    lstm = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(lstm) # input: [N, 21, 256] output: [N, 5, 256]
    # print(lstm.shape)
    lstm = layers.LSTM(64, dropout=0.3)(lstm) #input:[N, 5, 256] output:[N, 64] ##具体输出可能有点复杂, 可以直接看layers.LSTM函数
    x3 = tf.expand_dims(lstm, axis=1)       #input:[N, 64] output:[N, 1, 64]


    # x3 = layers.GlobalAveragePooling1D()(x)    #input:[N, 21, 256] output:[N, 256]
    # x3 = tf.expand_dims(x3, axis=1)             #input:[N, 256] ouput:[N, 1, 256]
    # x3 = layers.Concatenate(axis=-1)([lstm, x3]) #input:[N, 1, 64] [N, 1, 256] output:[N, 1, 320]
 
    # lstm = layers.Concatenate(axis=1)([x1, x2, x3]) 
    lstm = x3
    # lstm = layers.AveragePooling1D(pool_size=4, strides=4, padding='same')(lstm)

    x= layers.SeparableConv1D(256, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x= layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(x)
    global_pool = layers.GlobalAveragePooling1D()(x)    #input:[N, 21, 256] output:[N, 256]
    global_pool = tf.expand_dims(global_pool, axis=1)             #input:[N, 256] ouput:[N, 1, 256]
    global_pool = global_pool
    # global_pool = layers.Conv1D(64, kernel_size=1, strides=1, padding='same', use_bias=False)(global_pool)
    # global_pool = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(global_pool)
    # global_pool = layers.Activation('relu')(global_pool)
    concat_gloabal_lstm = layers.Concatenate(axis=-1)([lstm, global_pool]) #input:[N, 1, 64] [N, 1, 256] output:[N, 1, 320]
    
    # x = layers.AveragePooling1D(pool_size=2, strides=2, padding='same')(x)
    
    # x = layers.Concatenate(axis=1)([x, concat_gloabal_lstm])

    x = concat_gloabal_lstm
    x = ChannelAttention(320)(x)
    x = SpatialAttention()(x)

    ## fc layer
    x = layers.Dense(512, activation='relu')(x) #input:[N, 1, 320] output:[N, 1, 320]
    # x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x) #input:[N, 1, 320] output:[N, 1, 320]
    # x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x) #input:[N, 1, 320] output:[N, 320]
    x = layers.Dense(81, activation='softmax', name='psd')(x) #input:[N, 320] output:[N, 81]

    #predict = x
    model = keras.Model(inputs=input_dim, outputs=x)

    return model



# model = ChannelAttention(64)
# model = SpatialAttention()
model = model_simAM()
model.summary()
y = tf.random.normal(shape=(2, 81, 1))
x = model(y)
print(x.shape)

# y = tf.random.normal(shape=(2, 21, 256))
# # y = keras.initializers.zeros()(shape=(2, 21, 256))
# spp = spp()
# spp_reuslt = spp(y)
# print(spp_reuslt)

# sppf = sppf()
# sppf_reuslt = spp(y)
# print(sppf_reuslt)
