def plot_categories(df, var, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4)) 
    ax2 = ax.twinx()  
    ax.bar(df.index, df["%"], color='lightgrey')
    ax2.plot(df.index, df["Default"], color='green', label='Default')
    ax.axhline(y=0.01, color='red')
    ax.set_ylabel('Freq %')
    ax.set_xlabel(var)
    ax2.set_ylabel('Bad Rate')

    ax.set_xticks(df.index)
    ax.set_xticklabels(df[var], rotation=90)
  
def calculate_mean_target_per_category(df, var):
    total = len(df)
    temp_df = pd.Series(df[var].value_counts() / total).reset_index()
    temp_df.columns = [var, '%']
    temp_df = temp_df.merge(df.groupby([var])['Default'].mean().reset_index(),on=var,how='left')
    # Extraer límites inferiores de los intervalos como texto y convertirlos a números
    temp_df['order'] = temp_df[var].apply(lambda x: float(x.split(',')[0].replace('(', '').replace('inf', '1e10')))
    temp_df = temp_df.sort_values(by='order').reset_index(drop=True)
    temp_df = temp_df.drop(columns=['order']) 
    return temp_df

def calculate_mean_target_per_category(df, var):
    total = len(df)
    temp_df = pd.Series(df[var].value_counts() / total).reset_index()
    temp_df.columns = [var, '%']
    temp_df = temp_df.merge(df.groupby([var])['Default'].mean().reset_index(),on=var,how='left')
    # Extraer límites inferiores de los intervalos como texto y convertirlos a números
    temp_df['order'] = temp_df[var].apply(lambda x: float(x.split(',')[0].replace('(', '').replace('inf', '1e10')))
    temp_df = temp_df.sort_values(by='order').reset_index(drop=True)
    temp_df = temp_df.drop(columns=['order']) 
    return temp_df

def calcular_estadisticas(tabla, grupo, objetivo):
    grouped = tabla.groupby(grupo)
    total_count = grouped.size()
    good_count = grouped[objetivo].apply(lambda x: np.sum(x == 0))
    bad_count = grouped[objetivo].apply(lambda x: np.sum(x == 1))
    total_pct = total_count / np.sum(total_count)
    good_pct = good_count / np.sum(good_count)
    bad_pct = bad_count / np.sum(bad_count)
    bad_rate = bad_count / total_count
    good_pct_pct = good_pct / np.sum(good_pct)
    bad_pct_pct = bad_pct / np.sum(bad_pct)
    woe = np.where((bad_pct_pct == 0)|(good_pct_pct == 0), 0, np.log(good_pct_pct / bad_pct_pct))
    iv = (good_pct_pct - bad_pct_pct) * woe
    total_iv = np.sum(iv)
    result = pd.DataFrame({
        'Cantidad de deudores': total_count,
        'Distribucion de deudores_%': total_pct * 100,
        'Numero de deudores Buenos': good_count,
        'Numero de deudores Malos': bad_count,
        'Tasa de malos_%': bad_rate * 100,
        'Distribucion de buenos_%': good_pct * 100,
        'Distribucion de malos_%': bad_pct * 100,
        'WOE': woe,
        'IV_%': iv * 100,
        'Sum_IV_%': total_iv * 100
    })
    return result


def fun_graficadora(dataframes, column):
    """
    Grafica una columna específica para cada DataFrame en la lista de dataframes usando subplots.

    Parámetros:
    - dataframes: lista de tuplas (valor_q, DataFrame)
    - column: nombre de la columna que se va a graficar
    """
    n = len(dataframes)  # Número total de DataFrames
    cols = 3  # Número de columnas de subplots
    rows = (n // cols) + (n % cols > 0)  # Calculamos el número de filas necesarias para los subplots

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Creamos la figura con subplots
    axes = axes.flatten()  # Aplanamos el array de ejes para facilitar la iteración

    for i, (q_value, df) in enumerate(dataframes):  # Iteramos sobre los DataFrames y sus valores de q
        if column in df.columns:  # Verificamos que la columna exista en el DataFrame
            temp_df = calculate_mean_target_per_category(df, column)  # Calculamos las estadísticas necesarias
            temp_df2=calcular_estadisticas(df, column, 'Default')
            sum_iv_value = temp_df2['Sum_IV_%'].iloc[0]
            plot_categories(temp_df, column, ax=axes[i])  # Usamos la función personalizada para graficar
            axes[i].set_title(f'{column} - q={q_value} | IV_%: {sum_iv_value:.2f}')  # Añadimos el valor de Sum_IV_% al título
        else:
            print(f"Column '{column}' not found in DataFrame for q={q_value}.")
            axes[i].set_visible(False)  # Ocultamos el eje si la columna no está presente

    # Ocultamos cualquier subplot vacío
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()  # Ajustamos el layout para evitar solapamiento
    # Guarda la figura con el nombre de la columna
    plt.show()  # Mostramos la figura

    
# Creemos una función para identificar de manera masiva a las variables correlacionadas, en caso de tener un mayor volumen de variables
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                print(abs(corr_matrix.iloc[i, j]), corr_matrix.columns[i], corr_matrix.columns[j])
                colname = corr_matrix.columns[j]
                col_corr.add(colname)
    return col_corr



def getFeatureIV_Importance(df,features,target):
    featureIV_Importance=list()
    for v in features:
      iv, rep=calculate_woe_iv(df,v,target)
      featureIV_Importance.append(iv)
    display(pd.DataFrame({"Feature":features, "IV":featureIV_Importance}).sort_values("IV",ascending=False))


def calculate_woe_iv(dataset, feature_cat, target):
    lst = []
    feature=feature_cat
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return iv, dset

def Gini(y_true,y_pred,signo='+'):
    from sklearn.metrics import roc_auc_score
    return 2*roc_auc_score(y_true, y_pred)-1

# Una función para consolidar lo anterior
def plot_calibration_curve(y_true, probs, bins, strategy):

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, probs, n_bins=bins, strategy=strategy)

    max_val = max(mean_predicted_value)

    plt.figure(figsize=(8,10))
    plt.subplot(2, 1, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, label='Logistic Regression')
    plt.plot(np.linspace(0, max_val, bins), np.linspace(0, max_val, bins),
         linestyle='--', color='red', label='Perfect calibration')

    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of positive examples')
    plt.title('Calibration Curve')
    plt.legend(loc='upper left')


    plt.subplot(2, 1, 2)
    plt.hist(probs, range=(0, 1), bins=bins, density=True, stacked=True, alpha=0.3)
    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of examples')
    plt.title('Density')
    plt.show()
    
















