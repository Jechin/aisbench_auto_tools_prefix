# 数据集文件夹路径，需可访问(请使用绝对路径)
DATASET_PATH = "/root/aisbench_auto_tools_prefix/datasets"

# aisbench 工作路径, 为 git clone aisbench 后得到的 benchmark 目录的绝对路径
# 可通过命令 `pip show ais-bench-benchmark` 查询location
WORK_PATH = "/home/benchmark"

# 服务化配置的模型名称（与 vLLM/MindIE 部署名一致）
MODEL_NAME = "glm-5"
# 模型权重路径，用于读取 tokenizer（本地目录或 HuggingFace 模型 ID）
MODEL_PATH = "/home/weights/glm-5"
# 本地 MODEL_PATH 不存在时，回退使用该 HF 模型 ID 仅下载 tokenizer
TOKENIZER_FALLBACK = "zai-org/GLM-5"
# GLM-5 部署配置的 max_model_len（128k）
MAX_MODEL_LEN = 131072
# GLM 系列需 trust_remote_code
TRUST_REMOTE_CODE = True
# 请求目的 IP
HOST_IP = "141.xx.xx.xx"
# 请求目的端口
HOST_PORT = "8004"

## 鉴权信息
API_KEY = ""

# 如果使用稳态测试请将该字段设置为 "stable_stage"
DEFAULT_PERFORMANCE_TEST = "default_perf"

# aisbench输出日志保存路径
OUTPUT_DIR = "./outputs/default"

# 各节点信息，格式为 ["{ip}:{port}"]
# 用于查询vllm metrics计算各个dp域的prefix cache命中率，不配置默认为HOST_IP:HOST_PORT
# PD分离场景请填写各个节点的IP和对应dp域的port
# POD_INFO = ["141.xx.xx.11:8000","141.xx.xx.12:8000"]
POD_INFO = []
