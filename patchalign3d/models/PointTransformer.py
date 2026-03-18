import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from utils import get_missing_parameters_message, get_unexpected_parameters_message

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from collections import OrderedDict

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()  # type: ignore[union-attr]
    return fps_data

# class Group(nn.Module):
#     def __init__(self, num_group, group_size):
#         super().__init__()
#         self.num_group = num_group
#         self.group_size = group_size
#         self.knn = KNN(k=self.group_size, transpose_mode=True)


#     def forward(self, xyz):
#         '''
#             input: B N 3
#             ---------------------------
#             output: B G M 3
#             center : B G 3
#         '''
#         batch_size, num_points, _ = xyz.shape
#         # fps the centers out
#         center = fps(xyz, self.num_group) # B G 3
#         # knn to get the neighborhood
#         _, idx = self.knn(xyz, center) # B G M
#         assert idx.size(1) == self.num_group
#         assert idx.size(2) == self.group_size
#         idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
#         idx = idx + idx_base
#         idx = idx.view(-1)
#         neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
#         neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
#         # normalize
#         neighborhood = neighborhood - center.unsqueeze(2)
#         return neighborhood, center

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)


    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:, :, :3].contiguous()
            rgb = data[:, :, 3:].contiguous()
        else:
            xyz = xyz.contiguous()
        # fps the centers out
        center = fps(xyz.contiguous(), self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood_xyz = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.reshape(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.reshape(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center
    
class Encoder(nn.Module):
    def __init__(self, encoder_channel, color = False):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6 if color else 3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list

class TransformerEncoder_nopos(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list
    
from models.pointnet2_utils import PointNetFeaturePropagation
class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )
        assert new_combined_x is not None

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q




class get_model(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 
        self.color = config.color
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.num_classes = config.num_classes
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims, color = self.color)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,  # type: ignore[arg-type]
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3 + self.num_classes, mlp = [self.trans_dim * 4, self.trans_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

        self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        # ckpt = torch.load(bert_ckpt_path)
        # base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        # for k in list(base_ckpt.keys()):
        #     if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
        #         base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
        #     elif k.startswith('base_model'):
        #         base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
        #     del base_ckpt[k]

        # incompatible = self.load_state_dict(base_ckpt, strict=False)

        # if incompatible.missing_keys:
        #     print('missing_keys')
        #     print(
        #         get_missing_parameters_message(incompatible.missing_keys)
        #     )
        # if incompatible.unexpected_keys:
        #     print('unexpected_keys')
        #     print(
        #         get_unexpected_parameters_message(incompatible.unexpected_keys)
        #     )

        # print(f'[PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}')
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.point_encoder.'):
                state_dict[k.replace('module.point_encoder.', '')] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            # * print successful loading
            print("PointBERT's weights are successfully loaded from {}".format(bert_ckpt_path))

    def forward(self, pts, cls_label):
        B,_C,N = pts.shape

        # NEW: Isolate the 3D coordinates for all distance-based operations
        xyz = pts[:, :3, :].contiguous()
        xyz_transposed = xyz.transpose(-1, -2).contiguous() # Shape (B, N, 3) for FPS

        # The input to the main pipeline is still the 6D point cloud
        pts = pts.transpose(-1, -2) # B N 6
        
        # This part is correct. It takes 6D input but returns 3D centers.
        neighborhood, center = self.group_divider(pts)  # center shape is (B, G, 3)
        
        # Encoder and Transformer part
        group_input_tokens = self.encoder(neighborhood) 
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        
        # Positional embedding correctly uses the 3D center from the grouper
        pos = self.pos_embed(center)
        
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        feature_list = self.blocks(x, pos)
        feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]

        # --- Feature Propagation part needs correction ---
        cls_label_one_hot = cls_label.view(B, self.num_classes, 1).repeat(1, 1, N)
        
        # Level 0: The coordinates of the original full point cloud
        center_level_0 = xyz  # Use the 3-channel xyz tensor
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

        # CORRECTED: Perform FPS on the 3D coordinate tensor 'xyz_transposed'
        center_level_1 = fps(xyz_transposed, 512).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1
        center_level_2 = fps(xyz_transposed, 256).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2
        
        # Level 3: This was already correct because your Group class returns 3D centers
        center_level_3 = center.transpose(-1, -2).contiguous()

        # init the feature by 3nn propagation
        f_level_3 = feature_list[2]
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
        # Note: Propagating from level 2 to level 1, not level 3 to 1
        f_level_1 = self.propagation_1(center_level_1, center_level_2, f_level_1, feature_list[0]) 

        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # FC layers
        feat =  F.relu(self.bn1(self.conv1(f_level_0)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, f_level_3
    # def forward(self, pts, cls_label):
    #     B,C,N = pts.shape

    #     pts = pts.transpose(-1, -2) # B N 3
    #     # divide the point clo  ud in the same form. This is important
    #     neighborhood, center = self.group_divider(pts)
    #     # # generate mask
    #     # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
    #     # encoder the input cloud blocks
    #     group_input_tokens = self.encoder(neighborhood)  #  B G N
    #     group_input_tokens = self.reduce_dim(group_input_tokens)
    #     # prepare cls
    #     cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
    #     cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
    #     # add pos embedding
    #     pos = self.pos_embed(center)
    #     # final input
    #     x = torch.cat((cls_tokens, group_input_tokens), dim=1)
    #     pos = torch.cat((cls_pos, pos), dim=1)
    #     # transformer
    #     feature_list = self.blocks(x, pos)
    #     feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]

    #     cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
    #     center_level_0 = pts.transpose(-1, -2).contiguous()                     
    #     f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

    #     center_level_1 = fps(pts, 512).transpose(-1, -2).contiguous()            
    #     f_level_1 = center_level_1
    #     center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()            
    #     f_level_2 = center_level_2
    #     center_level_3 = center.transpose(-1, -2).contiguous()                 

    #     # init the feature by 3nn propagation
    #     f_level_3 = feature_list[2]
    #     f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
    #     f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])

    #     # bottom up
    #     f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
    #     f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
    #     f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

    #     # FC layers
    #     feat =  F.relu(self.bn1(self.conv1(f_level_0)))
    #     x = self.drop1(feat)
    #     x = self.conv2(x)
    #     x = F.log_softmax(x, dim=1)
    #     x = x.permute(0, 2, 1)
    #     return x, f_level_3
        
# class get_model(nn.Module):
#     def __init__(self, config, **kwargs):
#         super().__init__()
#         self.config = config

#         self.trans_dim = config.trans_dim
#         self.depth = config.depth 
#         self.drop_path_rate = config.drop_path_rate 
#         self.cls_dim = config.cls_dim 
#         self.num_heads = config.num_heads 
#         self.color = config.color
#         self.group_size = config.group_size
#         self.num_group = config.num_group
#         self.num_classes = config.num_classes
#         # grouper
#         self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
#         # define the encoder
#         self.encoder_dims =  config.encoder_dims
#         self.encoder = Encoder(encoder_channel = self.encoder_dims, color = self.color)
#         # bridge encoder and transformer
#         self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
#         self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

#         self.pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim)
#         )  

#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
#         self.blocks = TransformerEncoder(
#             embed_dim = self.trans_dim,
#             depth = self.depth,
#             drop_path_rate = dpr,
#             num_heads = self.num_heads
#         )

#         self.norm = nn.LayerNorm(self.trans_dim)

#         self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
#         self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
#         self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3 + self.num_classes, mlp = [self.trans_dim * 4, self.trans_dim])
#         self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
#         self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

#         self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)
#         self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

#         self.build_loss_func()
        
#     def build_loss_func(self):
#         self.loss_ce = nn.CrossEntropyLoss()
    
#     def get_loss_acc(self, ret, gt):
#         loss = self.loss_ce(ret, gt.long())
#         pred = ret.argmax(-1)
#         acc = (pred == gt).sum() / float(gt.size(0))
#         return loss, acc * 100

#     # ==================================================================
#     # === MODIFIED LOADING FUNCTION ====================================
#     # ==================================================================
#     def load_model_from_ckpt(self, bert_ckpt_path):
#         """
#         Loads weights selectively with enhanced logging.
#         - TRANSFERRED: Tokenizer (encoder, reduce_dim) and Positional Embeddings (pos_embed).
#         - FROM SCRATCH: Transformer blocks and Segmentation Head.
#         """
#         print("--- Loading pretrained weights for a custom experiment ---")
#         try:
#             ckpt = torch.load(bert_ckpt_path, map_location='cpu')
#         except Exception as e:
#             print(f"Error loading checkpoint file: {e}")
#             return

#         # Find the actual model state dictionary
#         if 'state_dict' in ckpt:
#             source_state_dict = ckpt['state_dict']
#         elif 'model' in ckpt:
#             source_state_dict = ckpt['model']
#         else:
#             source_state_dict = ckpt
        
#         state_dict_to_load = OrderedDict()
        
#         # Define the module prefixes we want to transfer from the pretrained model
#         prefixes_to_load = {
#             'encoder': 'module.point_encoder.encoder.',
#             'reduce_dim': 'module.point_encoder.reduce_dim.',
#             'pos_embed': 'module.point_encoder.pos_embed.'  # <-- ADDED
#         }
        
#         loaded_modules = set()

#         # Iterate through the source checkpoint keys
#         for k, v in source_state_dict.items():
#             # Check if the key corresponds to any module we want to load
#             for module_name, prefix in prefixes_to_load.items():
#                 if k.startswith(prefix):
#                     # Create the new key that matches our model's structure
#                     new_key = k.replace('module.point_encoder.', '')
#                     state_dict_to_load[new_key] = v
#                     loaded_modules.add(module_name)
#                     # Once matched, move to the next key in the checkpoint
#                     break 
        
#         # --- Enhanced Print Statements ---
#         print("\n--- Summary of Pretrained Weights ---")
#         for module in sorted(prefixes_to_load.keys()):
#             status = "FOUND in checkpoint" if module in loaded_modules else "NOT FOUND in checkpoint"
#             print(f"- Module '{module}': {status}")
#         print("-------------------------------------\n")

#         if not state_dict_to_load:
#             print("Warning: No weights were found in the checkpoint for the specified modules.")
#             print("The entire model will be trained from scratch.")
#             return

#         # Load the filtered state_dict. strict=False is essential
#         incompatible = self.load_state_dict(state_dict_to_load, strict=False)

#         if incompatible.missing_keys:
#             print("--- Missing Keys (as expected: Transformer blocks, Head, etc.) ---")
#             print(f"Total missing: {len(incompatible.missing_keys)}")
#             print(incompatible.missing_keys[:5], '...')
            
#         if incompatible.unexpected_keys:
#             print("\n--- Unexpected Keys (should be empty if filtering is correct) ---")
#             print(get_unexpected_parameters_message(incompatible.unexpected_keys))

#         print(f"\n[PointTransformer] Successfully loaded weights for: {', '.join(sorted(list(loaded_modules)))}")
        
#     def forward(self, pts, cls_label):
#         B,C,N = pts.shape

#         # NEW: Isolate the 3D coordinates for all distance-based operations
#         xyz = pts[:, :3, :].contiguous()
#         xyz_transposed = xyz.transpose(-1, -2).contiguous() # Shape (B, N, 3) for FPS

#         # The input to the main pipeline is still the 6D point cloud
#         pts = pts.transpose(-1, -2) # B N 6
        
#         # This part is correct. It takes 6D input but returns 3D centers.
#         neighborhood, center = self.group_divider(pts)  # center shape is (B, G, 3)
        
#         # Encoder and Transformer part
#         group_input_tokens = self.encoder(neighborhood) 
#         group_input_tokens = self.reduce_dim(group_input_tokens)

#         cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
#         cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        
#         # Positional embedding correctly uses the 3D center from the grouper
#         pos = self.pos_embed(center)
        
#         x = torch.cat((cls_tokens, group_input_tokens), dim=1)
#         pos = torch.cat((cls_pos, pos), dim=1)
#         feature_list = self.blocks(x, pos)
#         feature_list = [self.norm(x)[:,1:].transpose(-1, -2).contiguous() for x in feature_list]

#         # --- Feature Propagation part needs correction ---
#         cls_label_one_hot = cls_label.view(B, self.num_classes, 1).repeat(1, 1, N)
        
#         # Level 0: The coordinates of the original full point cloud
#         center_level_0 = xyz  # Use the 3-channel xyz tensor
#         f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)

#         # CORRECTED: Perform FPS on the 3D coordinate tensor 'xyz_transposed'
#         center_level_1 = fps(xyz_transposed, 512).transpose(-1, -2).contiguous()
#         f_level_1 = center_level_1
#         center_level_2 = fps(xyz_transposed, 256).transpose(-1, -2).contiguous()
#         f_level_2 = center_level_2
        
#         # Level 3: This was already correct because your Group class returns 3D centers
#         center_level_3 = center.transpose(-1, -2).contiguous()

#         # init the feature by 3nn propagation
#         f_level_3 = feature_list[2]
#         f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
#         # Note: Propagating from level 2 to level 1, not level 3 to 1
#         f_level_1 = self.propagation_1(center_level_1, center_level_2, f_level_1, feature_list[0]) 

#         # bottom up
#         f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
#         f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
#         f_level_0 =  self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

#         # FC layers
#         feat =  F.relu(self.bn1(self.conv1(f_level_0)))
#         x = self.drop1(feat)
#         x = self.conv2(x)
#         x = F.log_softmax(x, dim=1)
#         x = x.permute(0, 2, 1)
#         return x, f_level_3
    


# class get_model(nn.Module):
#     def __init__(self, config, **kwargs):
#         super().__init__()
#         self.config = config

#         self.trans_dim = config.trans_dim
#         self.cls_dim = config.cls_dim
#         self.color = config.color
#         self.group_size = config.group_size
#         self.num_group = config.num_group
#         self.num_classes = config.num_classes
#         # --- 1. Tokenizer Modules ---
#         self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
#         self.encoder_dims = config.encoder_dims
#         self.encoder = Encoder(encoder_channel=self.encoder_dims, color=self.color)
#         self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

#         # --- 2. Positional Embedding Module (RE-INTRODUCED) ---
#         self.pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim)
#         )

#         # --- 3. Decoder and Head Modules ---
#         self.propagation_2 = PointNetFeaturePropagation(in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 2, self.trans_dim])
#         self.propagation_1 = PointNetFeaturePropagation(in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 2, self.trans_dim])
#         self.propagation_0 = PointNetFeaturePropagation(in_channel=self.trans_dim + 3 + self.num_classes, mlp=[self.trans_dim * 2, self.trans_dim])
#         self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
#         self.dgcnn_pro_2 = DGCNN_Propagation(k=4)

#         self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.drop1 = nn.Dropout(0.5)
#         self.conv2 = nn.Conv1d(128, self.cls_dim, 1)

#         self.build_loss_func()

#     def build_loss_func(self):
#         self.loss_ce = nn.CrossEntropyLoss()

#     def get_loss_acc(self, ret, gt):
#         loss = self.loss_ce(ret, gt.long())
#         pred = ret.argmax(-1)
#         acc = (pred == gt).sum() / float(gt.size(0))
#         return loss, acc * 100

#     def load_model_from_ckpt(self, bert_ckpt_path):
#         """
#         Loads weights for the tokenizer AND the positional embeddings.
#         """
#         print("--- Loading weights for Tokenizer + Positional Embedding experiment ---")
#         try:
#             ckpt = torch.load(bert_ckpt_path, map_location='cpu')
#         except Exception as e:
#             print(f"Error loading checkpoint file: {e}")
#             return

#         if 'state_dict' in ckpt:
#             source_state_dict = ckpt['state_dict']
#         else:
#             source_state_dict = ckpt

#         state_dict_to_load = OrderedDict()

#         # --- MODIFIED: Added 'pos_embed' to the list of modules to load ---
#         prefixes_to_load = {
#             'encoder': 'module.point_encoder.encoder.',
#             'reduce_dim': 'module.point_encoder.reduce_dim.',
#             'pos_embed': 'module.point_encoder.pos_embed.'
#         }

#         loaded_modules = set()
#         for k, v in source_state_dict.items():
#             for module_name, prefix in prefixes_to_load.items():
#                 if k.startswith(prefix):
#                     new_key = k.replace('module.point_encoder.', '')
#                     state_dict_to_load[new_key] = v
#                     loaded_modules.add(module_name)
#                     break

#         print("\n--- Summary of Pretrained Weights ---")
#         for module in sorted(prefixes_to_load.keys()):
#             status = "FOUND in checkpoint" if module in loaded_modules else "NOT FOUND"
#             print(f"- Module '{module}': {status}")
#         print("-------------------------------------\n")

#         incompatible = self.load_state_dict(state_dict_to_load, strict=False)

#         if incompatible.missing_keys:
#             print(f"--- Missing Keys (as expected, Decoder and Head are from scratch) ---")
#             print(f"Total missing: {len(incompatible.missing_keys)}")

#         print(f"\n[PointTransformer] Successfully loaded weights for: {', '.join(sorted(list(loaded_modules)))}")


#     def forward(self, pts, cls_label):
#         B, C, N = pts.shape

#         xyz = pts[:, :3, :].contiguous()
#         xyz_transposed = xyz.transpose(-1, -2).contiguous()
#         pts = pts.transpose(-1, -2)

#         # --- 1. Tokenizer Stage ---
#         neighborhood, center = self.group_divider(pts)
#         group_input_tokens = self.encoder(neighborhood)
#         group_input_tokens = self.reduce_dim(group_input_tokens)  # Shape: (B, G, trans_dim)

#         # --- 2. Feature Enrichment Stage (NEW) ---
#         # Calculate positional embeddings from the group centers
#         pos = self.pos_embed(center) # Shape: (B, G, trans_dim)
#         # Add the positional information to the local features
#         enriched_tokens = group_input_tokens + pos

#         # --- 3. Decoder Stage ---
#         # The decoder now takes the "enriched_tokens" as its starting features.
#         cls_label_one_hot = cls_label.view(B, self.num_classes, 1).repeat(1, 1, N)

#         center_level_0 = xyz
#         f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)
#         center_level_1 = fps(xyz_transposed, 512).transpose(-1, -2).contiguous()
#         f_level_1 = center_level_1
#         center_level_2 = fps(xyz_transposed, 256).transpose(-1, -2).contiguous()
#         f_level_2 = center_level_2
#         center_level_3 = center.transpose(-1, -2).contiguous()

#         # The enriched features become the starting point for propagation
#         f_level_3 = enriched_tokens.transpose(-1, -2).contiguous()

#         # Propagate features up the hierarchy
#         f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, f_level_3)
#         f_level_1 = self.propagation_1(center_level_1, center_level_2, f_level_1, f_level_2)

#         # bottom up refinement
#         f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
#         f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
#         f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

#         # --- 4. Head Stage ---
#         feat = F.relu(self.bn1(self.conv1(f_level_0)))
#         x = self.drop1(feat)
#         x = self.conv2(x)
#         x = F.log_softmax(x, dim=1)
#         x = x.permute(0, 2, 1)

#         return x, f_level_3




class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
