from collections import defaultdict

class PosteriorOddsCalculator:
    def __init__(self, rules=None):

        self.default_rules = [
            {"conclusion": "SMIR", "premise": "RCS", "LS": 300, "LN": 1},
            {"conclusion": "SMIR", "premise": "RCAD", "LS": 75, "LN": 1},
            {"conclusion": "SMIR", "premise": "RCIB", "LS": 20, "LN": 1},
            {"conclusion": "SMIR", "premise": "RCVP", "LS": 4, "LN": 1},
            {"conclusion": "SMIRA", "premise": ["OR", "RCS", "RCAD", "RCIB", "RCVP"], "LS": None, "LN": None},
            {"conclusion": "SMIR", "premise": "SMIRA", "LS": 1, "LN": 0.0002},
            {"conclusion": "STIR", "premise": "FMGS", "LS": 2, "LN": 0.000001},
            {"conclusion": "FMGS&PT", "premise": ["AND", "FMGS", "PT"], "LS": None, "LN": None},
            {"conclusion": "STIR", "premise": "FMGS&PT", "LS": 100, "LN": 0.000001},
            {"conclusion": "HYPE", "premise": "SMIR", "LS": 300, "LN": 0.0001},
            {"conclusion": "HYPE", "premise": "STIR", "LS": 65, "LN": 0.01},
            {"conclusion": "FLE", "premise": "HYPE", "LS": 200, "LN": 0.0002},
            {"conclusion": "FLE", "premise": "CVR", "LS": 800, "LN": 1},
            {"conclusion": "FRE", "premise": "FLE", "LS": 5700, "LN": 0.0001},
            {"conclusion": "FRE", "premise": "OTFS", "LS": 5, "LN": 0.7}
        ]
        
        self.prior_probability = {
            "RCS": 0.001, 
            "RCAD": 0.001, 
            "RCIB": 0.001, 
            "RCVP": 0.001, 
            "SMIRA": 0.001, 
            "FMGS": 0.01, 
            "PT": 0.01, 
            "FMGS&PT": 0.01, 
            "STIR": 0.1,
            "SMIR": 0.03, 
            "HYPE": 0.01, 
            "CVR": 0.001, 
            "FLE": 0.005, 
            "OTFS": 0.1, 
            "FRE": 0.001
        }

        self.prior_odds = {k: i / (1 - i) for k, i in self.prior_probability.items()}

        self.rules = rules if rules is not None else self.default_rules
        
        self.rules_dict = defaultdict(list)
        for rule in self.rules:
            self.rules_dict[rule["conclusion"]].append(rule)
    
    def combine_odds(self, odds1, odds2, target):
        if odds1 is None:
            return odds2
        if odds2 is None:
            return odds1
        return odds1 * odds2 / self.prior_odds[target]
    
    def compute_expression(self, expr, odds_values, visited):
        if isinstance(expr, str):
            return self.calculate_odds(expr, odds_values, visited)
        elif isinstance(expr, list):
            op = expr[0]
            if op == "AND":
                return min(self.compute_expression(e, odds_values, visited) for e in expr[1:])
            elif op == "OR":
                return max(self.compute_expression(e, odds_values, visited) for e in expr[1:])
            else:
                raise ValueError(f"Unknown operator: {op}")
        else:
            raise ValueError("Invalid expression type")
    
    def calculate_odds(self, target, odds_values, visited):
        if target in odds_values:
            return odds_values[target]
        if target in visited:
            return 0.0
        
        visited.add(target)
        combined_odds = None
        
        for rule in self.rules_dict.get(target, []):
            premise_value = self.compute_expression(rule["premise"], odds_values, visited)
            if rule["LS"] is not None and rule["LN"] is not None:
                if premise_value == float('inf'):
                    contribution_probaility = rule["LS"] * self.prior_odds[target] / (1 + rule["LS"] * self.prior_odds[target])
                    contribution_odds = contribution_probaility / (1 - contribution_probaility)
                elif premise_value == 0:
                    contribution_probaility = rule["LN"] * self.prior_odds[target] / (1 + rule["LN"] * self.prior_odds[target])
                    contribution_odds = contribution_probaility / (1 - contribution_probaility)
                elif premise_value > self.prior_odds[rule["premise"]]:
                    contribution_probaility = self.prior_probability[target] + \
                    ((rule["LS"] * self.prior_odds[target] / (1 + rule["LS"] * self.prior_odds[target])) - self.prior_probability[target]) / \
                    (1 - self.prior_probability[rule["premise"]]) * ((premise_value / (1 + premise_value)) - self.prior_probability[rule["premise"]]) 
                    contribution_odds = contribution_probaility / (1 - contribution_probaility)
                elif premise_value <= self.prior_odds[rule["premise"]]:
                    contribution_probaility = (rule["LN"] * self.prior_odds[target] / (1 + rule["LN"] * self.prior_odds[target])) + \
                    ((self.prior_probability[target] - (rule["LN"] * self.prior_odds[target] / (1 + rule["LN"] * self.prior_odds[target]))) / \
                    self.prior_probability[rule["premise"]]) * (premise_value / (1 + premise_value))
                    contribution_odds = contribution_probaility / (1 - contribution_probaility)
            else:
                contribution_odds = premise_value 
            combined_odds = self.combine_odds(combined_odds, contribution_odds, target)
        
        odds_values[target] = combined_odds if combined_odds is not None else 0.0
        visited.remove(target)
        return odds_values[target]
    
    def calculate_posterior_odds(self, input_values, target="FRE"):
        odds_values = input_values.copy()
        self.calculate_odds(target, odds_values, set())
        return odds_values[target]

    def probability_to_odds(self, probability):
        if probability <= 0:
            return 0.0
        elif probability >= 1:
            return float('inf')
        else:
            return probability / (1 - probability)
        
    def confidence_to_odds(self, confidence, node):
        if confidence <= -5:
            return 0.0
        elif confidence >= 5:
            return float('inf')
        elif 0 <= confidence < 5:
            probability = confidence * (1 - self.prior_probability[node]) / 5 + self.prior_probability[node]
            return probability / (1 - probability)
        elif -5 < confidence < 0:
            probability = confidence * self.prior_probability[node] / 5 + self.prior_probability[node] 
            return probability / (1 - probability)

if __name__ == "__main__":
    calculator = PosteriorOddsCalculator()
    nodes = ["RCS", "RCAD", "RCIB", "RCVP", "FMGS", "PT", "CVR", "OTFS"]
    main_nodes = ["SMIR", "STIR", "HYPE", "FLE", "FRE"]
    input_values = {}
    print("Please enter the appropriate probability value")
    for node in nodes:
        while True:
            try:
                value_str = input(f"Please input {node}: ")
                value = float(value_str)
                if value < 0 or value > 1.0:
                    print("Error:The value must be between 0.0 and 1.0")
                    continue
                input_values[node] = calculator.probability_to_odds(value)
                break
            except ValueError:
                print("Error:Please enter a valid number")
    for node in main_nodes:
        posterior_odds = calculator.calculate_posterior_odds(input_values, node)
        posterior_probability = posterior_odds / (1 + posterior_odds)
        print(f"The posterior odds value of {node} is : {posterior_odds}\nThe posterior probability value of {node} is : {posterior_probability}")

    print("Please enter an appropriate confidence value")
    for node in nodes:
        while True:
            try:
                value_str = input(f"Please input {node}: ")
                value = float(value_str)
                if value < -5.0 or value > 5.0:
                    print("Error:The value must be between -5.0 and 5.0")
                    continue
                input_values[node] = calculator.confidence_to_odds(value, node)
                break
            except ValueError:
                print("Error:Please enter a valid number")

    for node in main_nodes:
        posterior_odds = calculator.calculate_posterior_odds(input_values, node)
        posterior_probability = posterior_odds / (1 + posterior_odds)
        print(f"The posterior odds value of {node} is : {posterior_odds}\nThe posterior probability value of {node} is : {posterior_probability}")