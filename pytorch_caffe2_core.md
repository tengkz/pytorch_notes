# 写在前面
在对Tensorflow的后端源码进行了拆解（参见[tensorflow源码解析系列文章索引](https://www.cnblogs.com/jicanghai/p/9589412.html)）之后，很想跟其它深度学习框架的实现进行对比，根据框架的流行程度，先选择了Pytorch。Pytorch的后端核心是直接复用了Caffe2，因此本文针对Caffe2源码的core模块进行了简单拆解。

# 目录
- 数据存储与表示
    - storage
    - tensor
    - blob
    - qtensor
- 操作
    - observer observable
    - operator
    - 操作求导
    - operator_schema
    - context
- 计算图
    - graph
    - net
    - transform
- 运行时
    - allocator
    - db
    - registry
    - module
    - scope_guard
    - workspace
    - init

# 1. 数据存储与表示
## 1.1 storage
Caffe2中对数据存储的最底层的描述是Storage，它实际上是指向StorageImpl的共享指针，后者包含数据类型、数据指针、容量、数据所在设备等信息。Storage的定义如下：
```
using Storage = std::shared_ptr<StorageImpl>;
class StorageImpl {
  public:
    //...
  protected:
    using DataPtr = std::shared_ptr<void>;
    int64_t capacity_ = 0;
    DataType data_type_;
    DataPtr data_ptr_;
    DeviceType device_type_ = CPU;
};
```
## 1.2 tensor
Caffe2中的数据统一使用Tensor表示，Tensor由TensorImpl实现，后者包含一个Storage。
```
graph LR
    Tensor-->|包含|TensorImpl
    TensorImpl-->|包含|Storage
    Storage-->|指向|StorageImpl
```

TensorImpl的定义如下：
```
class TensorImpl {
  public:
    //...
  protected:
    using DimVector = std::vector<TIndex>;
    DimVector dims_; //张量的维度
    TIndex size_ = -1; //张量中包含的元素数量
    Storage storage_; //底层存储
};
```
Tensor并非继承自TensorImpl，而是在内部包含了一个指向TensorImpl的指针，如下：
```
class Tensor final {
  protected:
    using TensorImplPtr = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
    TensorImplPtr impl_;
  //...
};
```
对Tensor的方法调用，通过重定向给TensorImpl实现。

## 1.3 blob
Blob是一个容器，包含了一个指针和这个指针指向内存的数据类型，在Caffe2中，大部分情况下Blob都包含一个指向Tensor的指针。
```
class Blob {
  public:
    //...
  private:
    TypeMeta meta_;
    void* pointer_ = nullptr;
    DestroyCall destroy_ = nullptr;
};
```

为了方便对Blob进行传输，定义了其序列化和反序列化的类，分别是BlobSerializerBase和BlobDeserializerBase，以及对应的为Tensor准备的序列化和反序列化类。
```
graph TB
    BlobSerializerBase-->|派生|TensorSerializer
    BlobDeserializerBase-->|派生|TensorDeserializer
```

## 1.4 qtensor
低精度的张量，为了便于快速进行低精度的整数乘法计算。具体的做法是，用更低的位数来表示整数，比如，用3个bit表示无符号整数，用4个bit表示有符号整数。低精度张量可以在略微损失模型精度的情况下，大大降低计算复杂度和存储空间大小。

# 操作
## 2.1 Observer Observable
Caffe2使用ObserverBase和Observable两个类实现了观察者模式。ObserverBase是基础观察器，用户可以通过继承此类创建新的观察器，而Observable是可被观察属性，用户可以通过继承此类获得可观察属性。

ObserverBase提供了观察器的统一接口，比较简单：
```
class ObserverBase {
  public:
    virtual void Start() {}
    virtual void Stop() {}
    T* subject() const {
        return subject_;
    }
  protected:
    T* subject_;
};
```
其中，subject_表示被观察对象的指针。

Observable封装了可被观察属性，内部包含了一个观察器的列表，结构如下：
```
class Observable {
  public:
    using Observer = ObserverBase<T>;
    const Observer* AttachObserver(std::unique_ptr<Observer> observer){} //添加观察器
    std::unique_ptr<Observer> DetachObserver(const Observer* observer_ptr){} //解除观察器
    virtual size_t NumObservers() {
        return num_observers_;
    } //观察器的数量
    void StartAllObservers(){} //启动所有观察器
    void StopAllObservers(){} //关闭所有观察器
  private:
    Observer* observer_cache_;
    size_t num_observers_ = 0;
  protected:
    std::vector<std::unique_ptr<Observer>> observer_list_; //观察器列表
};
```
## 2.2 Operator
Operator代表操作的具体实现，相当于Tensorflow中的kernel。Operator继承自OperatorBase，而后者继承自Observable，所以在Caffe2中，“操作”本质上是一个可观察的对象。
```
graph LR
    Observable-->|派生|OperatorBase
    OperatorBase-->|派生|Operator
```
OperatorBase类包含了操作需要的基本数据元素和接口：
```
class OperatorBase {
  private:
    Workspace* operator_ws_;
    std::shared_ptr<const OperatorDef> operator_def_;
    DeviceOption device_option_;
    std::string engine_;
    std::string type_;
    vector<const Blob*> inputs_;
    vector<Blob*> outputs_;
};
```
OperatorBase中包含了输入和输出的内存指针，可见，在Caffe2中，Operator本质上是一个运行时的对象，这与Tensorflow中Op的设计理念不同，在Tensorflow中，Op是一个编译时对象，仅规定了操作的类型和目标，并不包含具体数据，具体的计算实际上是通过Kernel完成的。

Operator继承自OperatorBase类：
```
class Operator : public OperatorBase {
  public:
    bool Run(int stream_id = 0) final {...}
    bool RunAsync(int stream_id = 0) final {...}
    virtual bool RunOnDevice() = 0;
};
```
实际上，Run和RunAsync最终都调用了RunOnDevice，完成实际的计算。

如果我们需要使用一些c10中定义的操作，需要将其转换为在Caffe2中可以调用的操作，可以通过如下的宏进行转换：
```
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(C10Add, C2MyAddOpName)
```
上述例子中，我们把一个C10Add操作，包装成C2MyAddOpName操作，供我们使用。为了实现这个功能，Caffe2还提供了一个包装类，C10OperatorWrapper。

## 2.3 操作求导
为了对操作求导，Caffe2推出了一个导数操作生成类，GradientMakerBase，方便用户定义对于某个操作的导数。类包含的数据成员如下：
```
//为密集和稀疏的blob提供统一的接口
struct GradientWrapper {
    string dense_;
    string indices_;
    string values_;
    inline bool IsDense(){}
    inline bool IsSparse(){}
    inline bool IsEmpty(){}
};
class GradientMakerBase {
  protected:
    const OperatorDef& def_;
    const vector<GradientWrapper>& g_output_;
    vector<GradientWrapper> g_input_;
};
```
可见，GradientMakerBase仅提供了输入输出，以及原操作。用户可以根据原操作，定制导数。

## 2.3 operator_schema
OpSchema是对操作的静态描述，相当于Tensorflow中的Op，包含的信息如下：
```
class OpSchema {
  private:
    string type_;
    string file_;
    string doc_;
    string onnx_schema_;
    std::vector<Argument> args_{};
    std::vector<std::pair<const char*, const char*>> input_desc_{};
    std::vector<std::pair<const char*, const char*>> output_desc_{};
    int line_ = 0;
    int min_input_ = 0;
    int max_input_ = std::numeric_limits<int>::max();
    int min_output_ = 0;
    int max_output_ = std::numeric_limits<int>::max();
    bool private_ = false;
    bool inputs_can_cross_devices_ = false;
    std::function<bool(int)> num_inputs_allowed = [](int) { return true; }
    std::function<bool(int)> num_outputs_allowed = [](int) { return true; }
    std::function<bool(int,int)> num_inputs_outputs_allowed_ = [](int,int) { return true; }
    std::function<int(int)> calculate_output_;
    std::function<bool(int,int)> inplace_allowed_ = [](int,int){}
    std::function<bool(int,int)> inplace_enforced_ = [](int,int){}
    TensorInferenceFunctionType tensor_inference_function_ = {...}
    std::unique_ptr<CostInferenceFunctionType> cost_inference_function_ = nullptr;
    DeviceInferenceFunctionType device_inference_function_ = {...}
};
```
另外Caffe2也提供了一个对于OpSchema的注册类OpSchemaRegistry，如下：
```
class OpSchemaRegistry {
  private:
    static CaffeMap<string, OpSchema>& map();
};
```

## 2.4 context
Caffe2中的context，其实就是Tensorflow中的OpKernelContext，为操作的实际计算提供通用的支持，主要包含内存拷贝的接口。所有实际的Context类必须继承自BaseContext，而Caffe2为我们准备了一个标准的Context接口，CPUContext类。另外，也同样为GPU准备了一个CUDAContext类。
```
graph LR
    BaseContext-->|派生|CPUContext
    BaseContext-->|派生|CUDAContext
```

# 3. 计算图
## 3.1 graph
Graph表示图的结构，图包含节点，节点包含操作。
```
graph LR
    Graph-->|包含|Node
    Node-->|包含|OperatorDef
```
Node包含的数据成员：
```
class Node {
  public:
    OperatorDef op;
    bool active = true; //操作是否被transformation删除
    std::map<int, std::vector<string>> parents;
    std::vector<int, std::vector<string>> children;
}
```
Graph包含的私有数据成员：
```
class Graph {
  private:
    NetDef netdef_;
    std::set<string> external_input_;
    std::set<string> external_output_;
    std::vector<Node> nodes_;
}
```

## 3.2 net
Net是一个可运行的Graph，包含了一个图的所有“操作”，以及它们的上下文。它继承自Observable，本质上是一个可观察的对象。数据成员如下：
```
class NetBase : public Observable<NetBase>{
  public:
    virtual bool Run(){...}
    virtual bool RunAsync();
  protected:
    vector<string> external_input_;
    vector<string> external_output_;
    string name_;
    vector<const Event*> events_;
    std::shared_ptr<const NetDef> net_def_;
};
```

NetBase派生出了三种子类，第一种是AsyncNetBase，它包含了异步执行网络所必须的数据和接口：
```
class AsyncNetBase : public NetBase {
  public:
    bool RunAsync() override;
  protected:
    bool canSchedule(...);
    std::vector<OperatorBase*> operators_;
    std::vector<dag_utils::OperatorNode> operator_nodes_;
    std::vector<std::vector<int>> chains_;
    std::vector<dag_utils::OpGraphNode> chain_nodes_;
    dag_utils::ExecutionChains execution_chains_;
};
```

第二种是SimpleNet，它表示了一种对图的单线程的顺序执行模式。
第三种是DAGNetBase，它表示了一种对图的多线程的dag执行模式。
相关的net类形成了一个继承体系：
```
graph TB
    Observable-->|派生|NetBase
    NetBase-->|派生|AsyncNetBase
    AsyncNetBase-->|派生|AsyncSchedulingNet
    NetBase-->|派生|DAGNetBase
    DAGNetBase-->|派生|DAGNet
    NetBase-->|派生|SimpleNet
    DAGNetBase-->|派生|AsyncDAGNet
    AsyncNetBase-->|派生|AsyncPollingNet
```

## 3.3 transform
transform是一种针对Caffe2的NetDef结构的操作，它将NetDef作为输入，输出新的经过变换的NetDef。它的工作步骤包括：
- 从旧的NetDef中构建一张图，这张图中保存了节点的连接信息；
- 在图中匹配指定的模式，找到它想要更改的子图；
- 用新的操作替换匹配到的子图；
- 根据图构建一个新的NetDef并返回；

Transform功能的实现，依赖于三个功能函数，如下：
- PatternRule（模式规则），它决定了对于一张子图和一个节点，是否可以将这个节点加入这个子图中；
- ValidatorRule（验证规则），它决定了一张子图是否是匹配的；
- ReplaceRule（替换规则），它对一张匹配的子图进行替换；

常用的模式如下：
- CONNECTED_SUBGRAPH，连接子图，它只能匹配连接的子图。比如对于图(1)-->(2)-->(3)-->(4)，它能够匹配到[2,3]和[4,3]，但不能匹配到[2,4]；
- SORTED_WRT_EXECUTION_ORDER，执行序模式，它只能匹配符合执行顺序的子图，节点之间不一定需要有连接，它比General模式要快，例如对于图(1)-->(2)-->(3)-->(4)，它可以匹配到[2,4],[3,4]，但不能匹配到[3,1]，[4,3]；
- GENERAL，它可以匹配到任何子图，比如，对于图(1)-->(2)-->(3)-->(4)来说，它可以匹配到子图[2,4]，[3,4]，[4,2,1]等；

# 4. 运行时
## 4.1 allocator
内存分配器。
```
graph TB
    CPUAllocator-->|派生|DefaultCPUAllocator
    CPUAllocator-->|派生|PinnedCPUAllocator
```

## 4.2 db
DB类是对kv存储的抽象。包含了用于读取DB数据的Cursor类，用于写DB数据的Transaction类，DB读取的包裹类DBReader，对DBReader进行序列化和反序列化的DBReaderSerializer和DBReaderDeserializer类。
```
graph TB
    DB-->|读数据时的游标类|Cursor
    DB-->|写数据时的事务类|Transaction
    DB-->|读数据包装|DBReader
    DBReader-->|序列化|DBReaderSerilizer
    DBReader-->|反序列化|DBReaderDeserilizer
```

## 4.3 registry
注册类，key为字符串，value可以为任意的类。结构如下：
```
class Registry {
  private:
    CaffeMap<SrcType, Creator> registry_;
    CaffeMap<SrcType, string> help_message_;
};
```

## 4.4 module
查看Caffe2已载入的模块，以及载入指定模块。模块指的是动态链接库。

## 4.5 scope_guard
是“初始化即资源获取”原语的实现，它保证了，如果不显式说明，函数的执行就会离开当前的scope。

## 4.6 workspace
Workspace包含了所有的运行时对象，包括blob和net，它是所有这些对象的拥有者，负责对这些对象进行管理。
```
class Workspace {
  private:
    typedef CaffeMap<string, unique_ptr<Blob>> BlobMap;
    BlobMap blob_map_;
    typedef CaffeMap<string, unique_ptr<NetBase>> NetMap;
    NetMap net_map_;
    const string root_folder_;
    const Workspace* shared_;
    std::unordered_map<string, std::pair<const Workspace*, string>> forwarded_blobs_;
    std::unique_ptr<ThreadPool> thread_pool_;
    std::mutex thread_pool_creation_mutex_;
    std::shared_ptr<Bookkeeper> bookkeeper_;
};
```

## 4.7 init
初始化整个Caffe2的运行环境，运行机制是，把需要在环境初始化中运行的函数注册到注册器中，初始化时，会在不同时期运行不同注册器中的函数。核心的函数如下：
```
CAFFE2_API bool GlobalInit(int* pargc, char*** argv);
```
整个初始化过程分为三步：
- 先运行通过REGISTER_CAFFE2_EARLY_INIT_FUNCTION注册的函数；
- 再解析Caffe的命令行参数，并启动日志记录系统；
- 最后运行通过REGISTER_CAFFE2_INIT_FUNCTION注册的函数；