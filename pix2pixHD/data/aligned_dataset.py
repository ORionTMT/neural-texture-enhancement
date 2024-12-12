import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        # A域图像（lofi）
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        print('A_paths:', self.A_paths)

        # B域图像（hifi）
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
        else:
            self.B_paths = []

        # depth图像
        self.dir_depth = os.path.join(opt.dataroot, opt.phase + '_depth')
        self.depth_paths = sorted(make_dataset(self.dir_depth))
        print('depth_paths:', self.depth_paths)

        # normal图像
        self.dir_normal = os.path.join(opt.dataroot, opt.phase + '_normal')
        self.normal_paths = sorted(make_dataset(self.dir_normal))
        print('normal_paths:', self.normal_paths)

        # material图像
        self.dir_material = os.path.join(opt.dataroot, opt.phase + '_material')
        self.material_paths = sorted(make_dataset(self.dir_material))
        print('material_paths:', self.material_paths)

        # uv图像
        self.dir_uv = os.path.join(opt.dataroot, opt.phase + '_uv')
        self.uv_paths = sorted(make_dataset(self.dir_uv))
        print('uv_paths:', self.uv_paths)

        # 是否使用instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))
        else:
            self.inst_paths = []

        # 是否使用features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        else:
            self.feat_paths = []

        self.dataset_size = len(self.A_paths)

      
    def __getitem__(self, index):        
        A_path = self.A_paths[index]              
        A = Image.open(A_path)

        # 获取变换参数
        params = get_params(self.opt, A.size)
        transform = get_transform(self.opt, params)  # 对A/B都使用同样的transform以保证对齐

        # 读取A域(lofi)
        A_tensor = transform(A.convert('RGB'))
        
        # 读取depth
        depth_path = self.depth_paths[index]
        depth_img = Image.open(depth_path).convert('RGB')
        depth_tensor = transform(depth_img)

        # 读取normal
        normal_path = self.normal_paths[index]
        normal_img = Image.open(normal_path).convert('RGB')
        normal_tensor = transform(normal_img)

        # 读取material
        material_path = self.material_paths[index]
        material_img = Image.open(material_path).convert('RGB')
        material_tensor = transform(material_img)

        # 读取uv
        uv_path = self.uv_paths[index]
        uv_img = Image.open(uv_path).convert('RGB')  # 若uv是2通道，需特殊处理，如将其补成3通道
        uv_tensor = transform(uv_img)

        # 将上述tensor拼接 (假设全是3通道)
        # 最终A端输入是 (3 lo-fi + 3 depth + 3 normal + 3 material + 3 uv) = 15通道
        A_tensor = torch.cat((A_tensor, depth_tensor, normal_tensor, material_tensor, uv_tensor), dim=0)

        B_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            B_tensor = transform(B)

        inst_tensor = feat_tensor = 0
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                    'feat': feat_tensor, 'path': A_path}

        return input_dict


    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'