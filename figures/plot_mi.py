
def plot_mis(csv_file, comparing_cols, title):
    df = pd.read_csv(csv_file)
    colors = ['b','g','r','c','m','y','k','w','burlywood']
    plt.figure()
    ax = plt.subplot(111) 
    for column, color in zip(comparing_cols, colors):
        col = df[column]
        ma = col.rolling(20).mean()
        mstd = col.rolling(20).std()
        plt.plot(ma.index, ma, color, label=column)
        plt.fill_between(mstd.index, ma - 2 * mstd, ma + 2 * mstd,
                     color=color, alpha=0.2)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Bits')
        ax.legend()
    plt.show()
