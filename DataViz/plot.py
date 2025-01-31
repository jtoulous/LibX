def get_column(df):
    valid_choice = 0
    X = None
    while valid_choice != 1:
        print('\nColumns:')
        for idx, col in enumerate(df.columns):
            print(f'  {idx} - {col}')

        chosen_idx = input('Choose idx: ')
        try:
            chosen_idx = int(chosen_idx)
        except Exception:
            print('Bad input, try again')   

        if chosen_idx < 0 or chosen_idx >= len(df.columns):
            print('Bad input, try again')   
        else:
            X = df[df.columns[chosen_idx]]
            valid_choice = 1
            df.drop(df.columns[chosen_idx], axis=1, inplace=True)
    return df, X


def scatter(X=None, Y=None, df=None, prompt=False, lbl=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    if X is not None:
        X = X.copy()

    if Y is not None:
        Y = Y.copy()

    if df is not None:
        df = df.copy()

    if prompt:
        df, X = get_column(df)
        df, Y = get_column(df)

    if lbl is None:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, color='blue', alpha=0.6, label='Data points')
        
        plt.xlabel(X.name)
        plt.ylabel(Y.name)
        plt.title('Scatter Plot')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
    else:
        labels = lbl.astype(str)

        unique_labels = labels.unique()
        colors = plt.cm.tab10(range(len(unique_labels)))
        label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(10, 6))
        for label in unique_labels:
            mask = labels == label
            plt.scatter(x[mask], y[mask], color=label_color_map[label], alpha=0.6, label=f'Label: {label}')
        
        plt.xlabel(X.name)
        plt.ylabel(Y.name)
        plt.title('Scatter Plot with Labels')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
    plt.show()



def hist(df, prompt=False, stats=True, bins=30):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    df = df.copy()
    if prompt:
        df, col = get_column(df)

    plt.figure(figsize=(10, 6))
    plt.hist(col, bins=bins, alpha=0.75, color='royalblue', edgecolor='black', linewidth=1.2)

    if stats:
        mean = np.mean(col)
        std = np.std(col)
        plt.axvline(mean, color='red', linestyle='-', linewidth=2, label='Moyenne')
        plt.axvline(mean - std, color='green', linestyle='--', linewidth=1.5, label='Moyenne - 1σ')
        plt.axvline(mean + std, color='green', linestyle='--', linewidth=1.5, label='Moyenne + 1σ')
        plt.legend()

    plt.title(f'Distribution de {col.name}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{col.name}', fontsize=12)
    plt.ylabel('Fréquence', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


def k_cluster(X=None, Y=None , df=None, prompt=False, stats=True, n_clusters=3):
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np

    if X is not None:
        X = X.copy()

    if Y is not None:
        Y = Y.copy()

    if df is not None:
        df_tmp = df.copy()

    if prompt:
        df_tmp, X = get_column(df_tmp)
        df_tmp, Y = get_column(df_tmp)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df[[X.name, Y.name]])

    labels = kmeans.labels_

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, c=labels, cmap='viridis', alpha=0.6)
    
    if stats:
        mean_X = X.mean()
        mean_Y = Y.mean()
        std_X = X.std()
        std_Y = Y.std()

        plt.axvline(mean_X, color='red', linestyle='--', label=f'Moyenne X: {mean_X:.2f}')
        plt.axhline(mean_Y, color='blue', linestyle='--', label=f'Moyenne Y: {mean_Y:.2f}')

        circle = plt.Circle((mean_X, mean_Y), radius=np.sqrt(std_X**2 + std_Y**2), color='green', fill=False, linestyle='--', label=f'Écart-type')
        plt.gca().add_artist(circle)

    plt.title('KMeans Clustering', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()


