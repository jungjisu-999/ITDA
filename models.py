import torch
import torch.nn as nn
from layers import InterAggregator

class CAREGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, relations_count):
        super(CAREGNN, self).__init__()
        self.aggregator = InterAggregator(input_dim, hidden_dim, relations_count)
        self.classifier = nn.Linear(hidden_dim, 1) # 사기 여부 판단

    def forward(self, x, adj_list):
        h = self.aggregator(x, adj_list)
        logits = self.classifier(h)
        # sigmoid를 제거하고 Raw Score(Logits)만 출력합니다.
        return logits