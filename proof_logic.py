import torch
from topos_ai.logic import SubobjectClassifier


def main():
    classifier = SubobjectClassifier()
    result_0 = classifier.logical_not(torch.tensor(0.0))
    result_1 = classifier.logical_not(torch.tensor(1.0))
    print(f"logical_not(0.0) = {result_0.item()}")
    print(f"logical_not(1.0) = {result_1.item()}")


if __name__ == "__main__":
    main()
