from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        print("\n Missing values count by column: ")
        missing_values = df.isna().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df: pd.DataFrame):
        print("\n Visualizing missing values: ")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isna(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()


if __name__ == "__main__":
    
    df = pd.read_csv("extracted_data/AmesHousing.csv")

    missing_values_analyzer = SimpleMissingValuesAnalysis()
    missing_values_analyzer.analyze(df)