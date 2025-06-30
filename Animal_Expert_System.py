class AnimalExpertSystem:
    def __init__(self):
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
        self.reverse_feature_mapping = {v: k for k, v in self.feature_names.items()}
        self.possible_animals = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']
        
        self.facts = set()
        self.reasoning_steps = []

    def forward_chain(self):
        """正向推理"""
        self.reasoning_steps.append("\n=== 开始正向推理 ===")
        new_fact_added = True
        while new_fact_added:
            new_fact_added = False
            for rule in self.rules:
                if all(c.startswith('M') for c in rule['conclusions']): # 只处理结论为中间节点（M开头）的规则，排除H开头的动物结论规则
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
        """反向推理"""
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

    def input_features(self, features_str):
        """输入特征处理"""
        self.facts.clear()
        self.reasoning_steps = []
        inputs = features_str.strip().split('，')
        
        self.reasoning_steps.append("=== 输入特征 ===")
        for item in inputs:
            code = self.reverse_feature_mapping.get(item)
            if code:
                self.facts.add(code)
                self.reasoning_steps.append(f"添加特征: {item}")
            else:
                self.reasoning_steps.append(f"忽略未知特征: {item}")
        self.reasoning_steps.append("")

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

if __name__ == "__main__":
    expert = AnimalExpertSystem()
    
    print("=== 动物识别专家系统 ===")
    print("请输入动物的特征代码（用逗号或空格分隔），可用的特征代码如下：")
    print("\n特征代码对照表：")
    feature_codes = {k: v for k, v in expert.feature_names.items() if k.startswith('F')}
    
    for i, (code, name) in enumerate(sorted(feature_codes.items())):
        print(f"{code}: {name}", end="\t")
        if (i + 1) % 4 == 0: 
            print()
    print("\n")
    
    print("请输入特征代码（例如：F1,F6,F12,F13）：")
    features_input = input()
    
    feature_codes = []
    for item in features_input.replace(',', ' ').split():
        item = item.strip().upper()  
        if item in expert.feature_names:
            feature_codes.append(item)
        else:
            print(f"警告：未知特征代码 '{item}'，已忽略")
    
    features_str = '，'.join([expert.feature_names[code] for code in feature_codes])
    
    print("\n您输入的特征：")
    for code in feature_codes:
        print(f"- {code}: {expert.feature_names[code]}")
    
    # 执行推理
    expert.input_features(features_str)
    results = expert.mixed_reasoning()
    
    # 输出结果
    expert.print_results(results)
    expert.print_reasoning_process()