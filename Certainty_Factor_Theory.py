from collections import defaultdict

class UncertaintyValueCalculator:
    def __init__(self, rules=None):
        """
        初始化不确定性值计算器
        
        parameter:
            rules: 可选的规则列表
        """
        self.default_rules = [
            {"conclusion": "H1", "premise": "E1", "cf": 0.9},
            {"conclusion": "H1", "premise": "E2", "cf": 0.8},
            {"conclusion": "H1", "premise": "E3", "cf": 0.9},
            {"conclusion": "E1", "premise": ["AND", "E4", "E5"], "cf": 0.9},
            {"conclusion": "E3", "premise": ["AND", "E6", ["OR", "E7", "E8"]], "cf": 1.0},
            {"conclusion": "H", "premise": "E9", "cf": 0.9},
            {"conclusion": "H", "premise": "H1", "cf": 0.9}
        ]
        
        self.rules = rules if rules is not None else self.default_rules
        
        self.rules_dict = defaultdict(list)
        for rule in self.rules:
            self.rules_dict[rule["conclusion"]].append(rule)
    
    def combine_cf(self, cf1, cf2):
        """
        后件相同之规则的结论的不确定性值的综合
        
        parameter:
            cf1: 第一个不确定性值
            cf2: 第二个不确定性值
            
        return:
            组合后的不确定性值
        """
        if cf1 is None:
            return cf2
        if cf2 is None:
            return cf1
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 - cf1 * cf2
        if cf1 <= 0 and cf2 <= 0:
            return cf1 + cf2 + cf1 * cf2    
        if cf1 * cf2 < 0 and abs(cf1 * cf2) == 1:
            return 0.0
        if cf1 * cf2 < 0 and abs(cf1 * cf2) != 1:
            tmp = 1 - min(abs(cf1), abs(cf2))
            return (cf1 + cf2) / tmp if tmp != 0 else 0.0
    
    def compute_expression(self, expr, cf_values, visited):
        """
        计算表达式的不确定性值
        
        parameter:
            expr: 表达式（字符串或列表）
            cf_values: 当前已知的不确定性值
            visited: 已访问节点集合
            
        return:
            表达式的不确定性值
        """
        if isinstance(expr, str):
            return self.calculate_cf(expr, cf_values, visited)
        elif isinstance(expr, list):
            op = expr[0]
            if op == "AND":
                return min(self.compute_expression(e, cf_values, visited) for e in expr[1:])
            elif op == "OR":
                return max(self.compute_expression(e, cf_values, visited) for e in expr[1:])
            else:
                raise ValueError(f"Unknown operator: {op}")
        else:
            raise ValueError("Invalid expression type")
    
    def calculate_cf(self, target, cf_values, visited):
        """
        计算目标节点的不确定性值
        
        parameter:
            target: 目标节点
            cf_values: 当前已知的不确定性值
            visited: 已访问节点集合
            
        return:
            目标节点的不确定性值
        """
        if target in cf_values:
            return cf_values[target]
        if target in visited:
            return 0.0
        
        visited.add(target)
        combined_cf = None
        
        for rule in self.rules_dict.get(target, []):
            premise_value = self.compute_expression(rule["premise"], cf_values, visited)
            contribution = premise_value * rule["cf"]
            combined_cf = self.combine_cf(combined_cf, contribution)
        
        cf_values[target] = combined_cf if combined_cf is not None else 0.0
        visited.remove(target)
        return cf_values[target]
    
    def calculate_uncertainty(self, input_values, target="H"):
        """
        计算目标节点的不确定性值
        
        parameter:
            input_values: 输入节点的不确定性值字典
            target: 目标节点，默认为"H"
            
        return:
            目标节点的不确定性值
        """
        cf_values = input_values.copy()
        self.calculate_cf(target, cf_values, set())
        return cf_values[target]


if __name__ == "__main__":
    calculator = UncertaintyValueCalculator()
    #题目输入E2:-0.8、E4:0.9、E5:0.8、E6:0.9、E7:-0.3、E8:0.8、E9:0.9
    nodes = ["E2", "E4", "E5", "E6", "E7", "E8", "E9"]
    input_values = {}
    for node in nodes:
        while True:
            try:
                value_str = input(f"Please input {node}: ")
                value = float(value_str)
                if value < -1.0 or value > 1.0:
                    print("Error:The value must be between -1.0 and 1.0")
                    continue
                input_values[node] = value
                break
            except ValueError:
                print("Error:Please enter a valid number")

    result = calculator.calculate_uncertainty(input_values)
    print(f"The uncertainty value of H is : {result}")