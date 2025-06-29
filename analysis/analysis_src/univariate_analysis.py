from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class UnivariateAnalysisStrategy(ABC):

    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):

    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):

    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=feature, palette='muted')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df, feature)


if __name__ == "__main__":

    df = pd.read_csv("extracted_data/AmesHousing.csv")

    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_analysis(df, "SalePrice")

    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_analysis(df, "Neighborhood")

    pass
