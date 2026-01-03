from AlgorithmImports import *

class ProbabilisticExecution(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        self.symbol = self.AddEquity("SPY", Resolution.Minute).Symbol
        self.SetWarmUp(30)

    def OnData(self, data):
        if self.IsWarmingUp:
            return

