# -------------------------------
# Rule-Based Expert System & Forward Chaining and Backward Chaining
# -------------------------------

class Rule:
    def __init__(self, premises, conclusion):
        self.premises = premises
        self.conclusion = conclusion


# ----------------------------
# Knowledge Base
# ----------------------------

rules = [
    Rule(["Fever", "Cough"], "Flu"),
    Rule(["Flu", "BodyPain"], "ViralInfection"),
    Rule(["ViralInfection"], "Medicine"),
    Rule(["Sneezing"], "Cold"),
    Rule(["Cold"], "Rest")
]


# ==================================================
# FORWARD CHAINING
# ==================================================

def forward_chaining(facts, rules):
    inferred = set(facts)
    print("\nInitial Facts:", inferred)
    changed = True

    while changed:
        changed = False
        for rule in rules:
            if all(p in inferred for p in rule.premises):
                if rule.conclusion not in inferred:
                    inferred.add(rule.conclusion)
                    print("Rule Fired:", rule.premises, "->", rule.conclusion)
                    changed = True

    return inferred


# ==================================================
# BACKWARD CHAINING
# ==================================================

def backward_chaining(goal, facts, rules, visited=None):
    if visited is None:
        visited = set()

    if goal in facts:
        print(goal, "is a known fact.")
        return True

    if goal in visited:
        return False

    visited.add(goal)

    for rule in rules:
        if rule.conclusion == goal:
            print("\nTrying Rule:", rule.premises, "->", rule.conclusion)
            success = True
            for premise in rule.premises:
                if not backward_chaining(premise, facts, rules, visited):
                    success = False
                    break
            if success:
                print("Goal", goal, "proved.")
                return True

    return False


# ==================================================
# Main Program
# ==================================================

facts = {"Fever", "Cough", "BodyPain"}

print("===== FORWARD CHAINING =====")
result = forward_chaining(facts, rules)

print("\nFinal Facts")
for fact in sorted(result):
    print(fact)

print("\n============================")

goal = "Medicine"

print("\n===== BACKWARD CHAINING =====")
if backward_chaining(goal, facts, rules):
    print("\nGoal Achieved:", goal)
else:
    print("\nGoal Cannot be Proved")