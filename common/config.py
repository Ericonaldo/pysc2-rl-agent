import json
import numpy as np
from s2clientprotocol import ui_pb2 as sc_ui
from s2clientprotocol import spatial_pb2 as sc_spatial
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, TYPES

# 设置要保存的feature信息

CAT = features.FeatureType.CATEGORICAL # 分类数据

# 动作函数的13个参数
DEFAULT_ARGS = dict(
    screen=0, # converts to (0,0)
    minimap=0,
    screen2=0,
    queued=0,#False,
    control_group_act=sc_ui.ActionControlGroup.Append,
    control_group_id=1,
    select_point_act=sc_spatial.ActionSpatialUnitSelectionPoint.Select,
    select_add=1,#True,
    select_unit_act=sc_ui.ActionMultiPanel.SelectAllOfType,
    select_unit_id=0,
    select_worker=sc_ui.ActionSelectIdleWorker.AddAll,
    build_queue_id=0,
    unload_id=0
)

# 结构化信息,原本minimap和screen是和其他信息都在一起的，作者把除了这二者以外的其他所有信息（没有子信息）都归纳为NON_SPATIAL_FEATURES，使其与minimap和screen平级，便于后续处理
# name => dims
NON_SPATIAL_FEATURES = dict(
    player=(11,),
    game_loop=(1,),
    score_cumulative=(13,),
    available_actions=(len(FUNCTIONS),), # (524,)
    single_select=(1, 7),
    # multi_select=(0, 7), # TODO
    # cargo=(0, 7), # TODO
    cargo_slots_available=(1,),
    # build_queue=(0, 7), # TODO
    control_groups=(10, 2),
)

class Config:
    # TODO extract embed_dim_fn to config
    def __init__(self, sz, map, run_id, embed_dim_fn=lambda x: max(1, round(np.log2(x))), restrict=False):
        self.run_id = run_id
        self.sz, self.map = sz, map
        self.embed_dim_fn = embed_dim_fn
        self.restrict = restrict
        self.feats = self.acts = self.act_args = self.arg_idx = self.ns_idx = None

    def build(self, cfg_path):
        feats, acts, act_args = self._load(cfg_path) # 加载参数，若参数不存在则保存

        if 'screen' not in feats:
            feats['screen'] = features.SCREEN_FEATURES._fields # ('height_map', 'visibility_map', 'creep', 'power', 'player_id', 'player_relative', 'unit_type', 'selected', 'unit_hit_points', 'unit_hit_points_ratio', 'unit_energy', 'unit_energy_ratio', 'unit_shields', 'unit_shields_ratio', 'unit_density', 'unit_density_aa', 'effects')
        if 'minimap' not in feats:
            feats['minimap'] = features.MINIMAP_FEATURES._fields # ('height_map', 'visibility_map', 'creep', 'camera', 'player_id', 'player_relative', 'selected')
        if 'non_spatial' not in feats:
            feats['non_spatial'] = NON_SPATIAL_FEATURES.keys()
        self.feats = feats

        # TODO not connected to anything atm
        if acts is None: # 动作函数
            acts = FUNCTIONS # <pysc2.lib.actions.Functions object at 0x7f74386f4160>
        self.acts = acts

        if act_args is None: # 动作函数的参数
            act_args = TYPES._fields # ('screen', 'minimap', 'screen2', 'queued', 'control_group_act', 'control_group_id', 'select_point_act', 'select_add', 'select_unit_act', 'select_unit_id', 'select_worker', 'build_queue_id', 'unload_id')
        self.act_args = act_args

        self.arg_idx = {arg: i for i, arg in enumerate(self.act_args)} #生成对应的字典，序号: 参数名
        self.ns_idx = {f: i for i, f in enumerate(self.feats['non_spatial'])} #生成对应的字典，参数名：序号

    def map_id(self):
        return self.map + str(self.sz)

    def full_id(self):
        if self.run_id == -1:
            return self.map_id()
        return self.map_id() + "/" + str(self.run_id)

    def policy_dims(self): # len(self.acts) 表示动作函数的个数为524， is_spatial()查询是否为空间动作参数，将两个部分拼接起来
        return [(len(self.acts), 0)] + [(getattr(TYPES, arg).sizes[0], is_spatial(arg)) for arg in self.act_args] # [(524, 0)] + [(0, True), (0, True), (0, True), (2, False), (5, False), (10, False), (4, False), (2, False), (4, False), (500, False), (4, False), (10, False), (500, False)]


    def screen_dims(self):
        return self._dims('screen')

    def minimap_dims(self):
        return self._dims('minimap') # dims是列表，表示每个feature map (sub info)的维度

    def non_spatial_dims(self): # 返回non_spatial_dims中每个信息的维度
        return [NON_SPATIAL_FEATURES[f] for f in self.feats['non_spatial']]

    # TODO maybe move preprocessing code into separate class?
    def preprocess(self, obs): # 预处理环境的所有观测量为需要的输入格式
        return [self._preprocess(obs, _type) for _type in ['screen', 'minimap'] + self.feats['non_spatial']]

    def _dims(self, _type):
        return [f.scale**(f.type == CAT) for f in self._feats(_type)] # 返回列表若是分类数据则维度为分类的类别，若是数值数据则是1，返回格式为[dim_subinfo1, dim_subinfo2, ...]

    def _feats(self, _type):
        feats = getattr(features, _type.upper() + '_FEATURES') # 获得features.*_FEATURES　(minimap, screen, non-spatial)
        return [getattr(feats, f_name) for f_name in self.feats[_type]] # 获得feats[*]中的数据，子图，返回格式为: [subinfo1, subinfo2, ...]

    def _preprocess(self, obs, _type): # 预处理环境的某个观测量为需要的输入格式
        if _type in self.feats['non_spatial']: #　如果是non_spatial的信息则返回列表，迭代其中的sub info，用_preprocess_non_spatial(self, ob, _type)处理
            return np.array([self._preprocess_non_spatial(ob, _type) for ob in obs])
        spatial = [[ob[_type][f.index] for f in self._feats(_type)] for ob in obs] # 将所有spatial的子图拼在一起, f.index是subinfo的索引, shape=[obs_dim, num_subfeatures, subfeatures_h, subfeatures_w]
        return np.array(spatial).transpose((0, 2, 3, 1)) # [obs_dim, subfeatures_h, subfeatures_w, num_subfeatures] 

    def _preprocess_non_spatial(self, ob, _type): # 预处理non_spatial信息，被上面调用
        if _type == 'available_actions':
            acts = np.zeros(len(self.acts))
            acts[ob['available_actions']] = 1
            return acts # 返回act,　可用的为1，不可用为0
        return ob[_type] # 返回观测到的信息

    def save(self, cfg_path):
        with open(cfg_path, 'w') as fl:
            json.dump({'feats': self.feats, 'act_args': self.act_args}, fl)

    def _load(self, cfg_path):
        with open(cfg_path, 'r') as fl:
            data = json.load(fl)
        return data.get('feats'), data.get('acts'), data.get('act_args')


def is_spatial(arg): # 判断动作信息是不是spatial的信息
    return arg in ['screen', 'screen2', 'minimap']
