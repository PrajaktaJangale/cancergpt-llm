import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "llm_agent",
    os.path.join(os.path.dirname(__file__), "04_llm_agent.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CancerGPTAgent = module.CancerGPTAgent