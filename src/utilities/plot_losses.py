import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(csv_file):

    df = pd.read_csv(csv_file)

    fig, ax1 = plt.subplots()

    sns.lineplot(data=df, x='step', y='val_loss', ax=ax1, label='Validation Loss', color='r', linewidth=2, linestyle='dashed')
    sns.lineplot(data=df, x='step', y='train_loss', ax=ax1, label='Training Loss', color='g', linewidth=2, linestyle='dashed')

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='step', y='val_accuracy', ax=ax2, label='Validation Accuracy', color='r', linewidth=2)
    sns.lineplot(data=df, x='step', y='train_accuracy', ax=ax2, label='Training Accuracy', color='g', linewidth=2)

    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='g')

    ax1.set_ylim(0, df['train_loss'].max()*1.1)
    ax2.set_ylim(0, 1)

    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right') 

    plt.title("Losses and Accuracy")

    plt.show()

plot_losses('logs/lightning_logs/version_19/metrics.csv')