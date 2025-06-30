import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class AnimalFeatureExtractor(nn.Module):
    def __init__(self, num_features=20):
        super(AnimalFeatureExtractor, self).__init__()
        # 使用预训练的ResNet18作为特征提取器
        self.base_model = models.resnet18(pretrained=True)
        # 去掉最后的全连接层
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        # 添加新的特征预测层
        self.feature_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.feature_predictor(x)
        return x

class AnimalExpertSystem:
    def __init__(self, model_path=None, threshold=0.7):
        self.rules = [
            {'conditions': ['F1'], 'conclusions': ['M1']},
            {'conditions': ['F2'], 'conclusions': ['M1']},
            {'conditions': ['F3'], 'conclusions': ['M4']},
            {'conditions': ['F4', 'F5'], 'conclusions': ['M4']},
            {'conditions': ['F6'], 'conclusions': ['M2']},
            {'conditions': ['F7', 'F8', 'F9'], 'conclusions': ['M2']},
            {'conditions': ['M1', 'F10'], 'conclusions': ['M3']},
            {'conditions': ['M1', 'F11'], 'conclusions': ['M3']},
            {'conditions': ['M1', 'M2', 'F12', 'F13'], 'conclusions': ['H1']},
            {'conditions': ['M1', 'M2', 'F12', 'F14'], 'conclusions': ['H2']},
            {'conditions': ['M3', 'F15', 'F16', 'F13'], 'conclusions': ['H3']},
            {'conditions': ['M3', 'F14'], 'conclusions': ['H4']},
            {'conditions': ['M4', 'F17', 'F15', 'F16', 'F18'], 'conclusions': ['H5']},
            {'conditions': ['M4', 'F17', 'F19', 'F18'], 'conclusions': ['H6']},
            {'conditions': ['M4', 'F20'], 'conclusions': ['H7']},
        ]
        for i, rule in enumerate(self.rules):
            rule['id'] = f'R{i+1}'

        self.feature_names = {
            'F1': '有毛发', 'F2': '有奶', 'F3': '有羽毛', 'F4': '会飞', 'F5': '下蛋',
            'F6': '吃肉', 'F7': '锋利牙齿', 'F8': '有爪', 'F9': '眼睛前视',
            'F10': '有蹄', 'F11': '反刍动物', 'F12': '黄褐色', 'F13': '暗斑', 
            'F14': '黑条纹', 'F15': '长脖子', 'F16': '长腿', 'F17': '不会飞',
            'F18': '黑白色', 'F19': '会游泳', 'F20': '善飞', 'M1': '哺乳动物',
            'M2': '食肉动物', 'M3': '有蹄类哺乳动物', 'M4': '鸟', 'H1': '豹',
            'H2': '虎', 'H3': '长颈鹿', 'H4': '斑马', 'H5': '鸵鸟', 'H6': '企鹅',
            'H7': '信天翁'
        }
        
        self.feature_codes = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 
                              'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20']
        
        self.reverse_feature_mapping = {v: k for k, v in self.feature_names.items()}
        self.possible_animals = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        self.model = AnimalFeatureExtractor(len(self.feature_codes))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 随具体数据集会进行改变
        ])
        
        self.threshold = threshold
        
        self.facts = set()
        self.reasoning_steps = []
        self.extracted_features = {}

    def extract_features_from_image(self, image_path):
        """从图片中提取特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(image_tensor).squeeze().numpy()
            
            # 应用阈值获取特征
            binary_features = features > self.threshold
            
            self.reasoning_steps.append("\n=== 图像特征提取 ===")
            self.facts.clear()
            self.extracted_features = {}
            
            for i, feature_code in enumerate(self.feature_codes):
                self.extracted_features[feature_code] = float(features[i])
                if binary_features[i]:
                    self.facts.add(feature_code)
                    self.reasoning_steps.append(f"检测到特征: {self.feature_names[feature_code]} (置信度: {features[i]:.2f})")
            
            self.reasoning_steps.append("=== 特征提取结束 ===\n")
            return True
        except Exception as e:
            self.reasoning_steps.append(f"图像处理错误: {str(e)}")
            return False

    def forward_chain(self):
        """正向推理链（添加日志记录）"""
        self.reasoning_steps.append("\n=== 开始正向推理 ===")
        new_fact_added = True
        while new_fact_added:
            new_fact_added = False
            for rule in self.rules:
                if all(c.startswith('M') for c in rule['conclusions']):  # 只处理结论为中间节点（M开头）的规则，排除H开头的动物结论规则
                    if all(cond in self.facts for cond in rule['conditions']):
                        for conclusion in rule['conclusions']:
                            if conclusion not in self.facts:
                                cond_str = ', '.join([self.feature_names[c] for c in rule['conditions']])
                                self.reasoning_steps.append(
                                    f"应用规则 {rule['id']}: 根据 {cond_str} "
                                    f"推导出 {self.feature_names[conclusion]}"
                                )
                                self.facts.add(conclusion)
                                new_fact_added = True
        self.reasoning_steps.append("=== 正向推理结束 ===\n")

    def backward_chain(self, animal, visited=None):
        """反向推理链"""
        visited = visited or set()
        animal_name = self.feature_names[animal]
        
        if animal in visited:
            self.reasoning_steps.append(f"检测到循环推导，终止推导 {animal_name}")
            return False
        visited.add(animal)
        
        if animal in self.facts:
            self.reasoning_steps.append(f"{animal_name} 已存在于事实库")
            return True

        self.reasoning_steps.append(f"\n开始推导 {animal_name}:")
        applicable_rules = [r for r in self.rules if animal in r['conclusions']]
        
        for rule in applicable_rules:
            cond_str = ' ∧ '.join([self.feature_names[c] for c in rule['conditions']])
            self.reasoning_steps.append(f"尝试规则 {rule['id']}: {cond_str} → {animal_name}")
            
            all_met = True
            for cond in rule['conditions']:
                # 检查特征是否已知
                if cond not in self.facts:
                    # 检查是否需要推导中间结论
                    if cond.startswith(('M', 'H')):
                        self.reasoning_steps.append(f"需要推导中间结论 {self.feature_names[cond]}")
                        if not self.backward_chain(cond, visited.copy()):
                            self.reasoning_steps.append(f"无法推导 {self.feature_names[cond]}")
                            all_met = False
                            break
                    else:
                        self.reasoning_steps.append(f"缺失必要特征: {self.feature_names[cond]}")
                        all_met = False
                        break
            
            if all_met:
                self.facts.add(animal)
                self.reasoning_steps.append(f"成功推导 {animal_name}!")
                return True
            else:
                self.reasoning_steps.append(f"规则 {rule['id']} 不适用")
        
        self.reasoning_steps.append(f"综上无法推导 {animal_name}")
        return False

    def reason_from_image(self, image_path):
        """基于图像的完整推理流程"""
        self.facts.clear()
        self.reasoning_steps = []
        
        # 从图像提取特征
        if not self.extract_features_from_image(image_path):
            return []
        
        # 执行推理
        return self.mixed_reasoning()

    def mixed_reasoning(self):
        """执行完整推理流程"""
        self.forward_chain()
        
        results = []
        self.reasoning_steps.append("\n=== 开始反向推理 ===")
        for animal in self.possible_animals:
            if self.backward_chain(animal):
                results.append(self.feature_names[animal])
        self.reasoning_steps.append("=== 反向推理结束 ===")
        
        return results

    def print_reasoning_process(self):
        """打印完整推理过程"""
        print("\n推理过程：")
        for step in self.reasoning_steps:
            print(step)

    def print_results(self, results):
        """输出结果"""
        if results:
            print("\n识别结果：")
            for animal in results:
                print(f"→ {animal}")
        else:
            print("\n无法确定动物类型")
    
    def train_model(self, data_folder, labels_file, epochs=10, batch_size=32, learning_rate=0.001):
        """训练深度学习模型"""
        # 示例训练流程
        # 实际应用中需要更详细的训练代码和数据处理
        
        print("开始训练模型...")
        
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()  
        
        # 数据加载和训练循环示例框架
        for epoch in range(epochs):
            running_loss = 0.0
            
            # 实际应用中应该使用DataLoader
            for i in range(100):  # 100个batch
                inputs = torch.randn(batch_size, 3, 224, 224)  # 示例输入
                labels = torch.rand(batch_size, len(self.feature_codes)) > 0.5  # 示例标签
                labels = labels.float()
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/100:.4f}")
        
        torch.save(self.model.state_dict(), "animal_feature_extractor.pth")
        print("模型训练完成并已保存")


def main():
    expert = AnimalExpertSystem(model_path="animal_feature_extractor.pth", threshold=0.6)
    
    while True:
        print("\n请选择操作：")
        print("1. 从图片识别动物")
        print("2. 训练深度学习模型")
        print("3. 退出")
        
        choice = input("请输入选项 (1-3): ")
        
        if choice == '1':
            image_path = input("请输入图片路径: ")
            if not os.path.exists(image_path):
                print(f"错误: 文件 '{image_path}' 不存在")
                continue
                
            results = expert.reason_from_image(image_path)
            expert.print_results(results)
            expert.print_reasoning_process()
            
        elif choice == '2':
            print("注意: 训练需要准备好的带标签数据集")
            data_folder = input("请输入数据集文件夹路径: ")
            labels_file = input("请输入标签文件路径: ")
            
            if not os.path.exists(data_folder) or not os.path.isdir(data_folder):
                print(f"错误: 文件夹 '{data_folder}' 不存在")
                continue
                
            if not os.path.exists(labels_file):
                print(f"错误: 文件 '{labels_file}' 不存在")
                continue
                
            epochs = int(input("请输入训练轮数 (默认10): ") or 10)
            expert.train_model(data_folder, labels_file, epochs=epochs)
            
        elif choice == '3':
            print("感谢使用，再见！")
            break
            
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()