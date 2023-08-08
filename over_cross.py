import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

def oversample_crossover(X, y, rows_1, mode="single", knn=False, random_state=1):
    '''Oversampled positively labeled data using a crossover
    operation. 

    Args: 
        X: Array of explanatory variables to oversample data from
        y: Array of labels associated with X
        rows_1: Number of positively labled rows required (original + oversampled)
        mode: Choice between single-point ("single"), two-point ("two"), and
            uniform ("uniform") crossover operation
        knn: If set to True, drops oversampled data whose nearest neighbor is not
            positively labeled.
        random_state: random state to pass to ensure reproducability of results

    Returns:
        X_crossover: Array of explanatory variables associated with the new 
            oversampled data (includes original + new data)
        y_crossover: Labels associated with X_crossover
    '''
    np.random.seed(random_state)

    # Potential because if the knn parameter is set to True,
    # those samples need to be checked if their nearest neighbor
    # has a label of 1
    potential_samples = []

    X_positive = X[y == 1]
    no_rows = X_positive.shape[0]
    no_cols = X_positive.shape[1]

    # assume % of 1s is at least 3%, this is relevant if knn=True
    for i in range(int(rows_1 / 0.03)):
        parent_1 = np.random.randint(0, no_rows)
        parent_2 = np.random.randint(0, no_rows)

        if mode == "single":
            cross_point = np.random.randint(1, no_cols)
            mask = np.array([1 if col_no < cross_point else 0
                             for col_no in range(no_cols)])

        elif mode == "two":
            cross_point_1 = np.random.randint(1, no_cols - 1)
            cross_point_2 = np.random.randint(cross_point_1, no_cols - 1)
            mask = np.array([
                1 if col_no < cross_point_1 or col_no > cross_point_2
                else 0 for col_no in range(no_cols)])

        elif mode == "uniform":
            mask = np.random.randint(0, 2, no_cols)

        else:
            raise ValueError("Accebtable options for mode: single, two, uniform")

        potential_samples.append(
            (X_positive[parent_1] * mask)
            + (X_positive[parent_2] * (1 - mask))
        )

    if knn == False:
        X_crossover = potential_samples
    else:
        scaler = MinMaxScaler().fit(X)
        X_scaled = scaler.transform(X)
        potential_samples_scaled = scaler.transform(potential_samples)

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_scaled, y)

        knn_filter = (model.predict_proba(
            potential_samples_scaled)[:, 1] > 0
        )

        X_crossover = np.array(potential_samples)[
            knn_filter]

    required_rows = rows_1 - (y == 1).sum()

    X_crossover = np.vstack([X, X_crossover[:required_rows]])

    y_crossover = np.hstack([
        y, np.ones(required_rows)])

    return X_crossover, y_crossover
