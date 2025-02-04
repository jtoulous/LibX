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


def get_stats(X, Y):
    import matplotlib.pyplot as plt
    import numpy as np
    
    mean_X = X.mean()
    mean_Y = Y.mean()
    std_X = X.std()
    std_Y = Y.std()

    plt.axvline(mean_X, color='red', linestyle='--', alpha=0.5, label=f'Moyenne X: {mean_X:.2f}')
    plt.axhline(mean_Y, color='red', linestyle='--', alpha=0.5, label=f'Moyenne Y: {mean_Y:.2f}')

    theta = np.linspace(0, 2 * np.pi, 100)
    radius = np.sqrt(std_X**2 + std_Y**2)
    circle_X = mean_X + radius * np.cos(theta)
    circle_Y = mean_Y + radius * np.sin(theta)

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_X = mean_X + std_X * np.cos(theta)
    circle_Y = mean_Y + std_Y * np.sin(theta)

    plt.plot(circle_X, circle_Y, color='blue', linestyle='--', alpha=0.5, label=f'Écart-type cercle')
    plt.fill(circle_X, circle_Y, color='blue', alpha=0.2)


def scatter(X=None, Y=None, df=None, lbl=None, stats=True):
    import pandas as pd
    import matplotlib.pyplot as plt

    if df is not None:
        df_tmp = df.copy()
        df_tmp, X = get_column(df_tmp)
        df_tmp, Y = get_column(df_tmp)
    else:
        X = X.copy()
        Y = Y.copy()

    if lbl is None:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, color='blue', alpha=0.6, label='Data points')
    else:
        labels = lbl.astype(str)
        unique_labels = labels.unique()
        colors = plt.cm.tab10(range(len(unique_labels)))
        label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(10, 6))
        for label in unique_labels:
            mask = labels == label
            plt.scatter(X[mask], Y[mask], color=label_color_map[label], alpha=0.6, label=f'Label: {label}')
        
    if stats:
        get_stats(X, Y)

    plt.xlabel(X.name)
    plt.ylabel(Y.name)
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

    if df is not None:
        df_tmp = df.copy()
        df_tmp, X = get_column(df_tmp)
        df_tmp, Y = get_column(df_tmp)
    else:
        X = X.copy()
        Y = Y.copy()

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(np.column_stack((X, Y)))
    labels = kmeans.labels_

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, c=labels, cmap='viridis', alpha=0.6)
    
    if stats:
        get_stats(X, Y)

    plt.title('KMeans Clustering', fontsize=14, fontweight='bold')
    plt.xlabel(f'{X.name}')
    plt.ylabel(f'{Y.name}')
    plt.legend()
    plt.colorbar()
    plt.show()


