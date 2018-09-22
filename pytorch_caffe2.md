# 写在前面
上一篇文章对Caffe2中的core模块进行了简单拆解[Caffe2源码解析之core](https://www.cnblogs.com/jicanghai/p/9689726.html)，回头一看其它模块也没多少代码，于是顺手都过了一遍，目的是大致了解每个模块的内容和目标，进一步理解Caffe2的整体框架。内容不多，略做整理如下。

# 目录
- core
- proto
    - caffe2.proto
    - hsm.proto
    - metanet.proto
- cuda_rtc
- db
- distributed
- ideep
- image
- mkl
- mobile
- mpi
- observers
- onnx
- operators
- opt
- perfkernels
- predictor
- queue
- sgd
- transform
- util
- python
- contrib

# core
参见[Caffe2源码解析之core](https://www.cnblogs.com/jicanghai/p/9689726.html)

# proto
包含了Caffe2中常用的protobuf定义，非常重要。我们按照所在文件进行介绍
## caffe2.proto
首先是TensorProto，它表示张量序列化后的结果，包括了张量的维度、数据类型、数据值、名称、所在设备等信息，如下：
```
message TensorProto {
    repeated int64 dims = 1;
    optional DataType data_type = 2 [default = FLOAT];
    repeated float float_data = 3 [packed = true];
    repeated int32 int32_data = 4 [packed = true];
    optional bytes byte_data = 5;
    repeated bytes string_data = 6;
    repeated double double_data = 9 [packed = true];
    repeated int64 int64_data = 10 [packed = true];
    optional string name = 7;
    
    //张量序列化之前所在的设备
    optional DeviceOption device_detail = 8;
    //张量在chunks中的位置
    message Segment {
        required int64 begin = 1;
        required int64 end = 2;
    }
    optional Segment segment = 11;
}
```

在core模块中讲到，Caffe2为了支持低精度的模型训练，设计了qtensor，当时没有详细介绍它的本质，实际上qtensor是对原张量进行了归一化，即减去bias再除以scale，然后对结果进行低精度表示，节省存储空间。因此在qtensor的序列化结果中，需要对归一化的参数进行记录，如下：
```
message QTensorProto {
    repeated int64 dims = 1;
    required int32 precision = 2;
    required double scale = 3;
    required double bias = 4;
    required bool is_signed = 5;
    repeated int32 data = 6 [packed = true];
    optional string name = 7;
    optional TensorProto.DataType data_type = 8 [default = INT32];
}
```

对于多个Tensor，可以使用TensorProto的复数版本，TensorProtos来存储。当然这只针对较小的张量，如果张量比较大，建议使用DB存储。

对于张量的形状，也有一个结构来表示，TensorShape。记得在Tensorflow中，对于张量形状的某些维度，在运行前可能并不是完全知道，因此这里在TensorShape的定义中，会添加参数对未知张量维度做处理。
```
message TensorShape {
    repeated int64 dims = 1;
    optional TensorProto.DataType data_type = 2 [default = FLOAT];
    repeated int32 unknown_dims = 3;
    optional bool unknown_shape = 4 [default = false];
    optional string name = 5;
}
```

参数用于对操作的描述（详见下文的OperatorDef定义），一个命名的参数要么包含单个的浮点型、整型或者字符串数据，要么包含上述类型的数组，如下：
```
message Argument {
    optional string name = 1;
    optional float f = 2;
    optional int64 i = 3;
    optional bytes s = 4;
    optional NetDef n = 8;
    repeated float floats = 5;
    repeated int64 ints = 6;
    repeated bytes strings = 7;
    repeated NetDef nets = 9;
}
```

目前Caffe2支持的设备类型：
```
enum DeviceType {
    CPU = 0;
    CUDA = 1;
    MKLDNN = 2;
    OPENGL = 3;
    OPENCL = 4;
    IDEEP = 5;
    HIP = 6;
    COMPILE_TIME_MAX_DEVICE_TYPES = 7;
    ONLY_FOR_TEST = 20901701;
}
```

目前Caffe2对于不同设备的描述proto还都是一致的，如果某个设备没有包含其中的某个字段，那么这个字段将被忽略。
```
message DeviceOption {
    optional int32 device_type = 1 [default = 0]; //0 is CPU
    optional int32 cuda_gpu_id = 2;
    optional uint32 random_seed = 3;
    optional string node_name = 4;
    optional int32 numa_node_id = 5;
    repeated string extra_info = 6;
    optional int32 hip_gpu_id = 7;
}
```

接下来是操作的定义：
```
message OperatorDef {
    repeated string input = 1; //输入的blob名称
    repeated string output = 2; //输出的blob名称
    optional string name = 3;
    optional string type = 4; //操作的类型，从操作注册器中创建操作对象时，需要这个信息
    optional string type = 4;
    repeated Argument arg = 5;
    optional DeviceOption device_option = 6; //操作运行所需要的设备
    
    //对于当前操作来说，如果对于指定的运行设备有多个计算引擎，这里可以指定一个具体的实现引擎。如果用户指定了一个引擎，但这个引擎在Caffe2的二进制包中并不存在，那么使用默认引擎
    optional string engine = 7;
    
    //控制输入，与Tensorflow中的控制输入类似，表达运行的先后顺序，而不是数据的输入。它仅在调度时被Net类使用
    repeated string control_input = 8;
    
    //is_gradient_op参数仅在形状推断（shape inference，与Tensorflow中类似）时使用，没有运行时的作用
    optional bool is_gradient_op = 9 [default = false];
    optional string debug_info = 10;
}
```

接下来NetDef的定义：
```
message NetDef {
    optional string name = 1;
    repeated OperatorDef op = 2;
    optional string type = 3; //network的执行方式，默认是simple
    optional DeviceOption device_option = 5; //整个net上所有操作的设备信息，在这里设置可以避免给每个操作单独设置
    repeated Argument arg = 6; //参数，包括num_workers，即当图被并行执行的时候，worker的数量
    
    repeated string external_input = 7;
    repeated string external_output = 8;
}
```

Caffe2中也可以像Tensorflow那样进行迭代计算，它使用了一个结构叫做ExecutionStep，如下：
```
message ExecutionStep {
    optional string name = 1;
    
    //ExecutionStep要么可以包含一个substep的集合，要么可以包含一些要运行的network的名称，但两者不能同时被设置
    repeated ExecutionStep substep = 2;
    repeated string network = 3;
    
    //当前的迭代需要运行的轮次，substeps和networks需要被顺序执行，每次执行被视为一轮迭代
    optional int64 num_iter = 4;
    
    //迭代执行结束的判断条件
    optional string criteria_network = 5;
    
    //如果这个字段被设置，那么就周期性的执行
    optional int64 run_every_ms = 11;
    
    //对于sub-steps，是顺序执行还是并行执行
    optional bool concurrent_substeps = 6;
    
    //一个用来判断当前执行是否需要终结的标志
    optional string should_stop_blob = 9;
    
    //如果为真，则当前执行仅执行一次，注意仅当should_stop_blob有效时才有效
    optional bool only_once = 10;
    
    //是否为当前执行构建一个子workspace
    optional bool create_workspace = 12;
    
    //子执行的并行度
    optional int32 num_concurrent_instances = 13;
}
```

如果说一个ExecutionStep是一次迭代执行，那么Plan就是一个完整的执行计划，后者包含前者：
```
message PlanDef {
    optional string name = 1;
    repeated NetDef netowrk = 2;
    repeated ExecutionStep execution_step = 3;
}
```

对于那些内部并不是Tensor的Blob，Caffe2定义了如下的结构：
```
message BlobProto {
    optional string name = 1;
    optional string type = 2;
    optional TensorProto tensor = 3;
    optional bytes content = 4;
    optional QTensorProto qtensor = 5;
    optional int32 content_num_chunks = 6;
    optional int32 content_chunk_id = 7;
}
```

最后，是对DBReader进行序列化的对象：
```
message DBReaderProto {
    optional string name = 1;
    optional string source = 2;
    optional string db_type = 3;
    optional string key = 4;
}
```

## hsm.proto
Word2Vec是早年Google提出的一个模型，目的是根据语料库获得词嵌入（embedding）。其中为了提高训练的速度提出了两种技术，一种是负采样（Negative Sampling），另外一种就是Hierarchical Softmax。因此，Caffe2专门设计了一个HSM操作，这个文件里包含的就是与之相关的proto，我们仅给出proto名称，含义比较显然：
```
message NodeProto;
message TreeProto;
message HierarchyProto;
message PathProto;
message PathNodeProto;
```

## metanet.proto
MetaNetDef，顾名思义，包含了NetDef的元数据。其结构如下：
```
message MetaNetDef {
    repeated BlobMap blobs = 1;
    repeated NetsMap nets = 2;
    optional ModelInfo modelInfo = 3;
    repeated PlanMap plans = 4;
    repeated StringMap applicationSpecificInfo = 5;
}
```
其中，对应的xxMap结构很简单，都是键值对，ModelInfo相对比较复杂，我们看下详细的定义：
```
message ModelInfo {
    optional string project = 1;
    optional string modelClass = 2;
    optional string version = 3;
    optional string predictorTtype = 4;
    optional string modelId = 5;
}
```

# cuda_rtc
cuda核生成相关的辅助代码。

# db
在Caffe2的执行过程中，需要重复使用和共享的参数，会被记录在一个db当中。在core模块中我们介绍过，db就是一个kv存储，这里包含了4种Caffe2中会用到的db，如下：
```
graph TB
    db-->|派生|LevelDB
    db-->|派生|LMDB
    db-->|派生|ProtoDB
    db-->|派生|ZmqDB
```

# distributed
Caffe2的分布式实现，依赖外部存储来保存共享的参数。常用的外部存储包括文件和redis。

外部存储的句柄用StoreHandler来表示，它包含了以下的核心API：
```
class StoreHandler {
  public:
    virtual void set(...) = 0;
    virtual std::string get(...) = 0;
    virtual int64_t add(...) = 0;
    virtual bool check(...) = 0;
    virtual void wait(...) = 0;
};
```
对应到计算图中，就有4个对store操作的op与之对应，如下：
```
graph TB
    Operator-->|派生|StoreSetOp
    Operator-->|派生|StoreGetOp
    Operator-->|派生|StoreAddOp
    Operator-->|派生|StoreWaitOp
```
刚才提到了，常用的存储方式为文件存储和redis存储，对应有两种存储句柄：
```
graph TB
    StoreHandler-->|派生|RedisStoreHandler
    StoreHandler-->|派生|FileStoreHandler
```
另外，还有两个创建存储的操作，如下：
```
graph TB
    Operator-->|派生|FileStoreHandlerCreateOp
    Operator-->|派生|RedisStoreHandler
```

# ideep
目前还不清楚具体含义。

# image
关于图像的操作，其中最重要的是对于图像读取的操作，ImageInputOp，它继承自PrefetchOperator，包含了图像读取的一系列功能。

# mkl
MKL全称是Intel Math Kernel Library，是英特尔提供的数学核心库，它对大量的数学过程进行了处理器级别的优化。这里包括了MKL相关的操作定义。注意，Tensorflow中也用到了MKL去优化数学运算，只不过它是在图优化的过程中，将MKL作为一种图优化遍历被引入，而Caffe2中将MKL直接融入到了操作内部。

# mobile
针对移动平台的特殊处理，具体还没看。

# mpi
Caffe2中的分布式计算，通过mpi实现。mpi的核心作用是在不同机器上的分布式进程中，进行数据传输和消息同步。针对mpi中的核心操作，比如Broadcast，Reduce等，Caffe2都给出了对应的操作来执行，具体如下：
```
graph TB
    Operator-->|派生|MPICreateCommonWorldOp
    Operator-->|派生|MPIBroadcastOp
    Operator-->|派生|MPIReduceOp
    Operator-->|派生|MPIAllgatherOp
    Operator-->|派生|MPIAllreduceOp
    Operator-->|派生|MPISendTensorOp
    Operator-->|派生|MPIReceiveTensorOp
```

# observers
给出了4种不同观察器的定义，如下：
- operator_attaching_net_observer，负责给net中的每一个operator添加观察器；
- profile_observer，负责对每个操作或整张图的执行消耗进行观察；
- runcnt_observer，负责对每个操作或者整张图的运行次数进行观察；
- time_observer，负责对每个操作或者整张图的运行时间进行观察；

# onnx
目前还不清楚。

# operators
操作的具体定义放在这里，代码量巨大，没来得及细看。

# opt
优化相关的类和函数，与Tensorflow一样，Caffe2也是通过对图遍历的方式实施优化，所有的优化遍历类必须继承自OptimizationPass，它的具体定义如下：
```
class OptimizationPass {
  public:
    OptimizationPass(NNModule* nn) : nn_(nn) {}
    virtual void run() = 0;
    virtual ~OptimizationPass(){}
    
  protected:
    NNModule* nn_;
};
```

# perfkernels
性能优化相关的kernel。

# predictor
一个predictor就是一个参数都确定好了的net。在深度学习中，我们通常会把待学习的模型表示为net，然后通过迭代的图计算，确定模型参数，将net转换为predictor。下面我们看下predictor的结构：
```
class Predictor {
  public:
    Predictor(const NetDef& init_net, const NetDef& run_net, Workspace* parent = nullptr, bool run_init = true, int optimization = 1);
    Predictor(PredictorConfig config);
    
    //以下是对()的重载，给定输入得到输出
    bool operator()(const TensorMap& inputs, TensorList* outputs);
    bool operator()(const TensorMap& inputs, TensorList* outputs);
    bool operator()(const TensorMap& inputs, TensorMap* outputs);
    
    const NetDef& def() const {
        return *config_.predict_net;
    };
    
    Workspace* ws(){
        return config_.ws.get();
    };
    const std::vector<std::string>& input_names() const {
        return config_.input_names;
    }
    const std::vector<std::string>& output_names() const {
        return config_.output_names;
    }
  private:
    bool run_map_workspace(const TensorMap& inputs);
    PredictorConfig config_;
};
```
其中，Predictor类最重要的一个私有数据成员是config_，我们看下PredictorConfig的定义：
```
struct PredictorConfig {
    std::shared_ptr<PredictorParameters> parameters;
    std::shared_ptr<NetDef> predict_net;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::string> parameter_names;
    std::shared_ptr<Workspace> ws;
};
```

# queue
先来看下BlobsQueue的定义：
```
class BlobsQueue : public std::enable_shared_from_this<BlobsQueue> {
  public:
    bool blockingRead(...);
    bool blockingWrite(...);
    void close();
  private:
    size_t numBlobs_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int64_t reader_{0};
    int64_t writer_{0};
    std::vector<std::vector<Blob*>> queue_; //核心队列数据
    const std::string name_;
};
```
注意看其中的数据元素queue_，它就是BlobsQueue的核心队列数据。

另外，BlobsQueue，也可以被看做是一种db，因此Caffe2定义了BlobQueueDB：
```
class BlobsQueueDB : public DB {
  public:
    BlobsQueueDB(...);
    void Close() override {}
    unique_ptr<Cursor> NetCursor() override{...}
    unique_ptr<Transaction> NewTransaction() override {...}
  private:
    std::shared_ptr<BlobsQueue> queue_;
    int key_blob_index_;
    int value_blob_index_;
    float timeout_secs_;
};
```

另外，Caffe2还针对BlobsQueue提出了提出了对队列进行处理的“操作”，把常用的队列处理方式，如入队、出队等，抽象为操作：
```
graph TB
    Operator-->|派生|CreateBlobsQueueOp
    Operator-->|派生|EnqueueBlobsOp
    Operator-->|派生|DequeueBlobsOp
    Operator-->|派生|CloseBlobsQueueOp
    Operator-->|派生|SafeEnqueueBlobsOp
    Operator-->|派生|SafeDequeueBlobsOp
    Operator-->|派生|WeightedSampleDequeueBlobsOp
```

另外，为了能支持一次多数据入队，Caffe2设计了RebatchingQueue类，它的简要结构如下：
```
class RebatchingQueue {
  public:
    bool enqueueOne(...);
    bool enqueueMany(...);
    bool dequeue(...);
  private:
    std::vector<std::vector<TensorCPU>> queue_;
};
```
与BlobsQueue最大的区别有两点，第一，核心数据queue_中存储的是TensorCPU而不是Blob*，第二，拥有EnqueueOne和EnqueueMany两种入队操作。

与BlobsQueue类似，Caffe2也为RebatchingQueue准备了对其进行处理的“操作”，与BlobsQueue类似，这里不再赘述。

# sgd
包含了与随机梯度下降有关的操作。基本上可以根据文件名猜测含义，这里仅列出文件名前缀，感兴趣的读者可以查阅源码：
```
adadelta_op
adagrad_op
adam_op
clip_tensor_op
fp16_momentum_sgd_op
fp32_momentum_sgd_op
ftrl_op
gftrl_op
iter_op
lars_op
learning_rate_adaption_op
learning_rate_functors
learning_rate_op
momentum_sgd_op
rmsprop_op
wngrad_op
yellowfin_op
```
有机会可以仔细研读下其中的细节。

# transform
根据core模块的内容我们知道，这里包含的是对图进行变换的方法。主要包括4种：
```
//公共子项消除，CSE，与Tensorflow类似
common_subexpression_elimination

//对卷积操作进行变换，提高效率
conv_to_nnpack_transform

//模式替换，允许你使用简单的接口定义模式替换规则，只需定义一模式子图和一个替换子图，在原图中寻找模式子图，然后替换为替换子图即可
pattern_net_transform

//单个操作的原地转换
single_op_transform
```
这些类形成了如下的继承体系：
```
graph TB
    Transform-->|派生|CommonSubexpressionEliminationTransform
    Transform-->|派生|SingleOpTransform
    Transform-->|派生|PatternNetTransform
    SingleOpTransform-->|派生|ConvToNNPackTransform
```

# util
应用类和函数，比较琐碎，暂时没有细看。

# python
通过前面的介绍我们了解到，Caffe2的核心代码是用"C++"实现的，为了方便在python中进行调用，需要一个工具，帮助python调用"C++"代码。这样的工具有很多，比如boost.python, swig，ctypes，pybind11等。Caffe2选择了pybind11，因为它对"C++"11支持的比较好，而且API比较简单。而Tensorflow中python前端调用"C++"后端使用的是swig，其实swig对"C++"11也能支持。两种设计选择的优劣目前的知识我们还不好评判。

具体的接口文件，是_import_c_extention.py，它首先会尝试载入gpu版本的Caffe2后端，如果失败了，会尝试载入CPU版本。其中，对于CPU后端的导入是通过如下的语句：
```
from caffe2.python.caffe2_pybind11_state import *
```
因此，在编译完成后，caffe2/python目录下会生成一个名为caffe2_pybind11_state.so的文件，是包含了Caffe2的"C++"后端的动态链接库，可以被python载入。

# contrib
同Tensorflow的contrib文件夹一样，包含了第三方贡献的、未正式加入Caffe2的模块，这里面大部分代码是用python开发的。随着版本迭代，经测试稳定后，这些模块会逐渐加入Caffe2的python模块。

# 写在后面
看过Tensorflow和Caffe2的核心代码之后，讲一讲自己的感受。
- 代码模块性，Tensorflow代码的模块性做的非常好，基础框架、运行时、图表示、图优化、op、kernel都区分的清清楚楚，而Caffe2的代码显得有些混杂，操作到处都是，给代码阅读带来了一点障碍。
- 代码规范性，Tensorflow代码的规范性要好很多，虽然核心代码是多个作者完成的，但代码风格非常统一，文件头的协议也非常一致。反观Caffe2的代码，协议混乱，代码风格不统一，东拼西凑的感觉比较强烈，代码在形式上的美感不足。
- 架构合理性，Tensorflow的野心很大，它的终极目标是变成一个全新的、面向数据流图计算的编程语言。这种编程语言基于op原语，利用op和kernel将编译期和运行期明确的区分开来，同时，它对于同一个数据流图的多线程并行执行机制，也像极了CPU流水线处理机制，因此，应该说，深度神经网络只是Tensorflow的一个副产品，它的真实价值远不止于此。反观Caffe2，很多设计有些短视了（比如用redis为中介做分布式执行），在提供更多灵活性的同时，也限制了它的高度。
当然，以上只是个人的一些猜测，随着理解的深入，我也会及时回来修正自己的观点，也欢迎大家来讨论。