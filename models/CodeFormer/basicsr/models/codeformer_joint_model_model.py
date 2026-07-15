# This wrapper ensures the CodeFormer joint model is registered with the basicsr model registry.
# It imports the actual implementation from codeformer_joint_model.py and exposes it under the expected module name
# ending with '_model.py' so that basicsr.models.__init__ will discover it.

from .codeformer_joint_model import CodeFormerJointModel
